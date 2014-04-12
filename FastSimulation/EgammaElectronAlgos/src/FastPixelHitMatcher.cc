// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      FastPixelHitMatcher
// 
/**\class FastPixelHitMatcher EgammaElectronAlgos/FastPixelHitMatcher

 Description: central class for finding compatible hits

 Implementation:
     <Notes on implementation>
*/
//
// Original Author: Patrick Janot
//
//

#include "FastPixelHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/Tracking/interface/TrackerRecHit.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define FAMOS_DEBUG

FastPixelHitMatcher::FastPixelHitMatcher(float ephi1min, float ephi1max, 
				     float pphi1min, float pphi1max, 
				     float phi2min, float phi2max, 
				     float z2minB, float z2maxB,
				     float r2minF, float r2maxF,
				     float rMinI, float rMaxI, 
				     bool searchInTIDTEC) :
  ephi1min(ephi1min), ephi1max(ephi1max), 
  pphi1min(pphi1min), pphi1max(pphi1max), 
  phi2min(phi2min), phi2max(phi2max), 
  z2minB(z2minB), z2maxB(z2maxB), 
  r2minF(r2minF), r2maxF(r2maxF),
  rMinI(rMinI), rMaxI(rMaxI), 
  searchInTIDTEC(searchInTIDTEC),
  theTrackerGeometry(0),
  theMagneticField(0),
  theGeomSearchTracker(0),
  _theGeometry(0),
  thePixelLayers(50,static_cast<TrackerLayer*>(0)),
  vertex(0.) {}

FastPixelHitMatcher::~FastPixelHitMatcher() { }

void 
FastPixelHitMatcher::setES(const MagneticFieldMap* aFieldMap, 
			 const TrackerGeometry* aTrackerGeometry, 
			 const GeometricSearchTracker* geomSearchTracker,
			 const TrackerInteractionGeometry* interactionGeometry) {
  
  // initialize the tracker geometry and the magnetic field map
  theTrackerGeometry = aTrackerGeometry; 
  //theMagneticField = aMagField;
  theGeomSearchTracker = geomSearchTracker;
  _theGeometry = interactionGeometry;
  theFieldMap = aFieldMap;
  
  // Initialize (if not already done) the simplified magnetic field geometry
  // MagneticFieldMap::instance( theMagneticField, _theGeometry );
  
  // The pixel layers in the simplified geometry 
  unsigned layer = 1;
  std::list<TrackerLayer>::const_iterator cyliter = _theGeometry->cylinderBegin();
  for ( ; cyliter != _theGeometry->cylinderEnd() ; ++cyliter ) {
    if ( layer != cyliter->layerNumber() ) continue;
    thePixelLayers[layer++] = &(*cyliter);
  }
  
}

std::vector< std::pair<FastPixelHitMatcher::ConstRecHitPointer, 
		       FastPixelHitMatcher::ConstRecHitPointer> > 
FastPixelHitMatcher::compatibleHits(const GlobalPoint& thePos,
				  const GlobalPoint& theVertex,
				  float energy,
				  std::vector<TrackerRecHit>& theHits) { 
  
  std::vector< std::pair<FastPixelHitMatcher::ConstRecHitPointer, 
    FastPixelHitMatcher::ConstRecHitPointer> > result;
#ifdef FAMOS_DEBUG
  std::cout << "[FastPixelHitMatcher::compatibleHits] entering .. " << std::endl;
#endif
  
  double zCluster = thePos.z();
  double rCluster = thePos.perp();
  
  // The cluster inferred energy-momentum
  double theLength = thePos.mag();
  XYZTLorentzVector theMom(thePos.x(), thePos.y(), thePos.z(), theLength);
  theMom *= energy / theLength;
  XYZTLorentzVector theVert(thePos.x(),thePos.y(),thePos.z(),0.);
  XYZTLorentzVector theNominalVertex(theVertex.x(), theVertex.y(), theVertex.z(), 0.);
  
  // The corresponding RawParticles (to be propagated for e- and e+
  ParticlePropagator myElec(theMom,theVert,-1.,theFieldMap);
  ParticlePropagator myPosi(theMom,theVert,+1.,theFieldMap); 
#ifdef FAMOS_DEBUG
  std::cout << "elec/posi before propagation " << std::endl << myElec << std::endl << myPosi << std::endl;
#endif
  
  // Propagate the e- and the e+ hypothesis to the nominal vertex
  // by modifying the pT direction in an appropriate manner.
  myElec.propagateToNominalVertex(theNominalVertex);
  myPosi.propagateToNominalVertex(theNominalVertex);
#ifdef FAMOS_DEBUG
  std::cout << "elec/posi after propagation " << std::endl << myElec << std::endl << myPosi << std::endl;
#endif
  
  // Look for an appropriate see in the pixel detector
  bool thereIsASeed = false;
  unsigned nHits = theHits.size();
  
  for ( unsigned firstHit=0; firstHit<nHits-1; ++firstHit ) { 
    for ( unsigned secondHit=firstHit+1; secondHit<nHits; ++secondHit ) {      

      // Is there a seed associated to this pair of Pixel hits?
      thereIsASeed = isASeed(myElec,myPosi,theVertex,
			     rCluster,zCluster,
			     theHits[firstHit],
			     theHits[secondHit]);

#ifdef FAMOS_DEBUG
      std::cout << "Is there a seed with hits " << firstHit << " & "<< secondHit << "? " << thereIsASeed << std::endl;
#endif
      if ( !thereIsASeed ) continue;
      
      ConstRecHitPointer theFirstHit = 
	GenericTransientTrackingRecHit::build(theHits[firstHit].geomDet(),
					      theHits[firstHit].hit());
      ConstRecHitPointer theSecondHit = 
	GenericTransientTrackingRecHit::build(theHits[secondHit].geomDet(),
					      theHits[secondHit].hit());
      result.push_back(std::pair<
		       FastPixelHitMatcher::ConstRecHitPointer,
		       FastPixelHitMatcher::ConstRecHitPointer>(theFirstHit,theSecondHit));
      
    }
  }
  
  return result;
}

bool FastPixelHitMatcher::isASeed(const ParticlePropagator& myElec,
				const ParticlePropagator& myPosi,
				const GlobalPoint& theVertex,
				double rCluster,
				double zCluster,
				const TrackerRecHit& hit1,
				const TrackerRecHit& hit2) {
  
  // Check that the two hits are not on the same layer
  if ( hit2.isOnTheSameLayer(hit1) ) return false;
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: The two hits are not on the same layer - OK " << std::endl;
#endif

  // Check that the first hit is on PXB or PXD
  if ( hit1.subDetId() > 2 ) return false;
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: The first hits is on the pixel detector " << std::endl;
#endif

  // Impose the track to originate from zVertex = 0. and check the 
  // compatibility with the first hit (beam spot constraint)
  GlobalPoint firstHit = hit1.globalPosition();
  bool rzok = checkRZCompatibility(zCluster, rCluster, 0., z1min, z1max, firstHit, hit1.subDetId()>1);
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: rzok (1) = " << rzok << std::endl;
#endif
  if ( !rzok ) return false;
  
  // Refine the Z vertex by imposing the track to pass through the first RecHit, 
  // and check the compatibility with the second rechit 
  GlobalPoint secondHit = hit2.globalPosition(); 
  rzok = false;

  // The orgin Z vertex for thet track passing through the first rechit
  vertex = zVertex(zCluster, rCluster, firstHit);

  // Compute R (forward) or Z (barrel) predicted for the second hit and check compatibility
  if ( hit2.subDetId() == 1 ) {
    rzok = checkRZCompatibility(zCluster, rCluster, vertex, z2minB, z2maxB, secondHit, false);
  } else if ( hit2.subDetId() == 2 ) {  
    rzok = checkRZCompatibility(zCluster, rCluster, vertex, r2minF, r2maxF, secondHit, true);
  } else { 
    rzok = checkRZCompatibility(zCluster, rCluster, vertex, rMinI, rMaxI, secondHit, true);
  }
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: rzok (2) = " << rzok << std::endl;
#endif
  if ( !rzok ) return false;

  // Propagate the inferred electron (positron) to the first layer,
  // check the compatibility with the first hit, and propagate back
  // to the nominal vertex with the hit constraint
  ParticlePropagator elec(myElec);
  ParticlePropagator posi(myPosi);
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: elec1 to be propagated to first layer" << std::endl;
#endif
  bool elec1 = propagateToLayer(elec,theVertex,firstHit,
				ephi1min,ephi1max,hit1.cylinderNumber());
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: posi1 to be propagated to first layer" << std::endl;
#endif
  bool posi1 = propagateToLayer(posi,theVertex,firstHit,
				pphi1min,pphi1max,hit1.cylinderNumber());

#ifdef FAMOS_DEBUG
  std::cout << "isASeed: elec1 / posi1 " << elec1 << " " << posi1 << std::endl;
#endif
  // Neither the electron not the positron hypothesis work...
  if ( !elec1 && !posi1 ) return false;
  
  // Otherwise, propagate to the second layer, check the compatibility
  // with the second hit and propagate back to the nominal vertex with 
  // the hit constraint
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: elec2 to be propagated to second layer" << std::endl;
#endif
  bool elec2 = elec1 && propagateToLayer(elec,theVertex,secondHit,
					 phi2min,phi2max,hit2.cylinderNumber());
  
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: posi2 to be propagated to second layer" << std::endl;
#endif
  bool posi2 = posi1 && propagateToLayer(posi,theVertex,secondHit,
					 phi2min,phi2max,hit2.cylinderNumber());
  
#ifdef FAMOS_DEBUG
  std::cout << "isASeed: elec2 / posi2 " << elec2 << " " << posi2 << std::endl;
#endif
  if ( !elec2 && !posi2 ) return false;

  return true;

}
  
bool 
FastPixelHitMatcher::propagateToLayer(ParticlePropagator& myPart,
				    const GlobalPoint& theVertex,
				    GlobalPoint& theHit,
				    double phimin, double phimax,
				    unsigned layer) {

  // Set the z position of the particle to the predicted one
  XYZTLorentzVector theNominalVertex(theVertex.x(), theVertex.y(), vertex, 0.);
  myPart.setVertex(theNominalVertex);
#ifdef FAMOS_DEBUG
  std::cout << "propagateToLayer: propagateToLayer: Before propagation (0) " << myPart << std::endl;
#endif

  // Propagate the inferred electron (positron) to the first layer
  // Use the radius (barrel) or the z (forward) of the hit instead 
  // of the inaccurate layer radius and z.
  double rCyl = thePixelLayers[layer]->forward() ? 999. : theHit.perp();
  double zCyl = thePixelLayers[layer]->forward() ? fabs(theHit.z()) : 999.;
  BaseParticlePropagator* myBasePart = (BaseParticlePropagator*)(&myPart);
  myBasePart->setPropagationConditions(rCyl,zCyl);
  // myPart.setPropagationConditions(*(thePixelLayers[layer]));

  // bool success = myPart.propagateToBoundSurface(*(thePixelLayers[layer]));
  bool success = myPart.propagate();
#ifdef FAMOS_DEBUG
  std::cout << "propagateToLayer: Success ? " << success << std::endl;
  std::cout << "propagateToLayer: After  propagation (1) " << myPart << std::endl;
  std::cout << "propagateToLayer: The hit               " << theHit << std::endl; 
#endif
      
  // Check that propagated particle is within the proper phi range.
  if ( success ) {
    double dphi = myPart.vertex().phi() - theHit.phi();
    if ( dphi < -M_PI ) 
      dphi = dphi + 2.*M_PI;
    else if ( dphi > M_PI ) 
      dphi = dphi - 2.*M_PI;
#ifdef FAMOS_DEBUG
    std::cout << "propagateToLayer: Phi range ? " << phimin << " < " << dphi << " < " << phimax << std::endl; 
#endif
    if ( dphi < phimin || dphi > phimax ) success = false;
  }
      
  // Impose the track to go through the hit and propagate back to 
  // the nominal vertex
  if ( success ) {
    myPart.setVertex( XYZTLorentzVector(theHit.x(), theHit.y(), theHit.z(), 0.) );
    myPart.propagateToNominalVertex(theNominalVertex);
#ifdef FAMOS_DEBUG
    std::cout << "propagateToLayer: After  propagation (2) " << myPart << std::endl;
#endif
  }

  return success;

}
					 

double 
FastPixelHitMatcher::zVertex(double zCluster,
			   double rCluster,
			   GlobalPoint& theHit)
{

  // Refine the Z vertex by imposing the track to pass through the RecHit
  double pxHitz = theHit.z();
  double pxHitr = theHit.perp();
  return pxHitz - pxHitr*(zCluster-pxHitz)/(rCluster-pxHitr);

}


bool
FastPixelHitMatcher::checkRZCompatibility(double zCluster,double rCluster, 
					double zVertex, 
					float rzMin, float rzMax,
					GlobalPoint& theHit, 
					bool forward) 
{

  // The hit position
  double zHit = theHit.z();
  double rHit = theHit.perp();

  // Compute the intersection of a line joining the cluster position 
  // and the predicted z origin (zVertex) with the layer hit 
  // (returns R for a forward layer, and Z for a barrel layer)
  double checkRZ = forward ?
    (zHit-zVertex)/(zCluster-zVertex) * rCluster
    :
    zVertex + rHit * (zCluster-zVertex)/rCluster;

  // This value is then compared to check with the actual hit position 
  // (in R for a forward layer, in Z for a barrel layer)

  return forward ?
    checkRZ+rzMin < rHit && rHit < checkRZ+rzMax 
    :
    checkRZ+rzMin < zHit && zHit < checkRZ+rzMax;
    
}







