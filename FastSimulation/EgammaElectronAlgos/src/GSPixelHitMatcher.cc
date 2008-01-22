// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      GSPixelHitMatcher
// 
/**\class GSPixelHitMatcher EgammaElectronAlgos/GSPixelHitMatcher

 Description: central class for finding compatible hits

 Implementation:
     <Notes on implementation>
*/
//
// Original Author: Patrick Janot
//
//

#include "FastSimulation/EgammaElectronAlgos/interface/GSPixelHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include "FastSimulation/TrackerSetup/interface/TrackerLayer.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

GSPixelHitMatcher::GSPixelHitMatcher(float ephi1min, float ephi1max, 
				     float pphi1min, float pphi1max, 
				     float phi2min, float phi2max, 
				     float z1min, float z1max, 
				     float z2min, float z2max) 
    :
    ephi1min(ephi1min), ephi1max(ephi1max), 
    pphi1min(pphi1min), pphi1max(pphi1max), 
    phi2min(phi2min), phi2max(phi2max), 
    z1min(z1min), z1max(z1max), 
    z2min(z2min), z2max(z2max), 
    theTrackerGeometry(0),
    theMagneticField(0),
    theGeomSearchTracker(0),
    _theGeometry(0),
    thePixelLayers(6,static_cast<TrackerLayer*>(0)),
    vertex(0.) {}

GSPixelHitMatcher::~GSPixelHitMatcher() { }

void 
GSPixelHitMatcher::setES(const MagneticFieldMap* aFieldMap, 
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
    thePixelLayers.push_back(&(*cyliter));
    thePixelLayers[layer] = &(*cyliter);
    if ( layer++ == 5 ) break;
  }
  
}

std::vector< std::pair<GSPixelHitMatcher::ConstRecHitPointer, 
		       GSPixelHitMatcher::ConstRecHitPointer> > 
GSPixelHitMatcher::compatibleHits(const GlobalPoint& thePos,
				  const GlobalPoint& theVertex,
				  float energy,
				  std::vector<ConstRecHitPointer>& thePixelRecHits) { 

  std::vector< std::pair<GSPixelHitMatcher::ConstRecHitPointer, 
                         GSPixelHitMatcher::ConstRecHitPointer> > result;
  LogDebug("") << "[GSPixelHitMatcher::compatibleHits] entering .. ";

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
  
  // Propagate the e- and the e+ hypothesis to the nominal vertex
  // by modifying the pT direction in an appropriate manner.
  myElec.propagateToNominalVertex(theNominalVertex);
  myPosi.propagateToNominalVertex(theNominalVertex);
  
  // Look for an appropriate see in the pixel detector
  bool thereIsASeed = false;
  unsigned nHits = thePixelRecHits.size();
  
  for ( unsigned firstHit=0; firstHit<nHits-1; ++firstHit ) { 
    for ( unsigned secondHit=firstHit+1; secondHit<nHits; ++secondHit ) {      

      // Is there a seed associated to this pair of Pixel hits?
      thereIsASeed = isASeed(myElec,myPosi,
			     rCluster,zCluster,
			     thePixelRecHits[firstHit],
			     thePixelRecHits[secondHit]);
      if ( !thereIsASeed ) continue;

      result.push_back(std::pair<GSPixelHitMatcher::ConstRecHitPointer,
		                 GSPixelHitMatcher::ConstRecHitPointer>
		                   (thePixelRecHits[firstHit],
		                    thePixelRecHits[secondHit]));
      
    }
  }
  
  return result;
}

bool GSPixelHitMatcher::isASeed(const ParticlePropagator& myElec,
				const ParticlePropagator& myPosi,
				double rCluster,
				double zCluster,
				ConstRecHitPointer hit1,
				ConstRecHitPointer hit2) {
  
  // Check that the two hits are not on the same layer
  unsigned firstHitLayer, secondHitLayer;
  // First hit:
  const DetId& detId1 = hit1->geographicalId();
  unsigned int subdetId1 = detId1.subdetId(); 
  if ( subdetId1 ==  PixelSubdetector::PixelBarrel ) { 
    PXBDetId pxbid1(detId1.rawId()); 
    firstHitLayer = pxbid1.layer();  
  } else if ( subdetId1 ==  PixelSubdetector::PixelEndcap ) { 
    PXFDetId pxfid1(detId1.rawId()); 
    firstHitLayer = pxfid1.disk()+3;
  } else {
    firstHitLayer = 0;
    std::cout << "Warning !!! This pixel hit is neither PXB nor PXF" << std::endl;
  }

  // Second hit
  const DetId& detId2 = hit2->geographicalId();
  unsigned int subdetId2 = detId2.subdetId(); 
  if ( subdetId2 ==  PixelSubdetector::PixelBarrel ) { 
    PXBDetId pxbid2(detId2.rawId()); 
    secondHitLayer = pxbid2.layer();  
  } else if ( subdetId2 ==  PixelSubdetector::PixelEndcap ) { 
    PXFDetId pxfid2(detId2.rawId()); 
    secondHitLayer = pxfid2.disk()+3;
  } else {
    secondHitLayer = 0;
    std::cout << "Warning !!! This pixel hit is neither PXB nor PXF" << std::endl;
  }

  if ( firstHitLayer == secondHitLayer ) return false;

  // Refine the Z vertex by imposing the track to pass 
  // through the first RecHit, and check compatibility
  const GeomDet* geomDet1( theTrackerGeometry->idToDet(detId1) );
  GlobalPoint firstHit = geomDet1->surface().toGlobal(hit1->localPosition());
  double zVertexPred = zVertex(zCluster, rCluster, firstHit);
  bool z1ok = zCompatible(zVertexPred,0.,z1min,z1max,firstHitLayer<4);
  if ( !z1ok ) return false;
  
  // Do the same with the second RecHit ...
  const GeomDet* geomDet2( theTrackerGeometry->idToDet(detId2) );
  GlobalPoint secondHit = geomDet2->surface().toGlobal(hit2->localPosition());
  double zVertexPred2 = zVertex(zCluster, rCluster, secondHit);
  bool z2ok = zCompatible(zVertexPred2,zVertexPred,z2min,z2max,secondHitLayer<4);
  vertex = zVertexPred2;
  if ( !z2ok ) return false; 

  // Propagate the inferred electron (positron) to the first layer,
  // check the compatibility with the first hit, and propagate back
  // to the nominal vertex with the hit constraint
  ParticlePropagator elec(myElec);
  ParticlePropagator posi(myPosi);
  bool elec1 = propagateToLayer(elec,firstHit,zVertexPred,
				ephi1min,ephi1max,firstHitLayer);
  bool posi1 = propagateToLayer(posi,firstHit,zVertexPred,
				pphi1min,pphi1max,firstHitLayer);
  
  // Neither the electron not the positron hypothesis work...
  if ( !elec1 && !posi1 ) return false;
  
  // Otherwise, propagate to the second layer, check the compatibility
  // with the second hit and propagate back to the nominal vertex with 
  // the hit constraint
  bool elec2 = elec1 && propagateToLayer(elec,secondHit,zVertexPred2,
					 phi2min,phi2max,secondHitLayer);
  
  bool posi2 = posi1 && propagateToLayer(posi,secondHit,zVertexPred2,
					 phi2min,phi2max,secondHitLayer);
  
  if ( !elec2 && !posi2 ) return false;

  return true;

}
  
bool 
GSPixelHitMatcher::propagateToLayer(ParticlePropagator& myPart,
				    GlobalPoint& theHit,
				    double zVertex,
				    double phimin, double phimax,
				    unsigned layer) {

  // Set the z position of the particle to the predicted one
  myPart.setVertex( XYZTLorentzVector(0.,0.,zVertex,0.) );

  // Propagate the inferred electron (positron) to the first layer
  myPart.setPropagationConditions(*(thePixelLayers[layer]));

  bool success = myPart.propagateToBoundSurface(*(thePixelLayers[layer]));
      
  // Check that propagated particle is within the proper phi range.
  if ( success ) {
    double dphi = myPart.vertex().phi() - theHit.phi();
    if ( dphi < -M_PI ) 
      dphi = dphi + 2.*M_PI;
    else if ( dphi > M_PI ) 
      dphi = dphi - 2.*M_PI;
    if ( dphi < phimin || dphi > phimax ) success = false;
  }
      
  // Impose the track to go through the hit and propagate back to 
  // the nominal vertex
  if ( success ) {
    myPart.setVertex( XYZTLorentzVector(theHit.x(), theHit.y(), theHit.z(), 0.) );
    myPart.propagateToNominalVertex();
  }

  return success;

}
					 

double 
GSPixelHitMatcher::zVertex(double zCluster,
			   double rCluster,
			   GlobalPoint& theHit)
{

  // Refine the Z vertex by imposing the track to pass through the RecHit
  double pxHitz = theHit.z();
  double pxHitr = theHit.perp();
  return pxHitz - pxHitr*(zCluster-pxHitz)/(rCluster-pxHitr);

}

bool
GSPixelHitMatcher::zCompatible(double zVertex, double zPrior, 
			       double zmin, double zmax,
			       bool barrel) 
{
  
  bool success = true;

  double deltaZ = zVertex - zPrior;

  // Check the z compatibility with the prior hypothesis
  // Double the tolerance in the forward layer
  if ( barrel ) {
    if ( deltaZ > zmax || deltaZ < zmin ) success = false;
  }
  else {
    if ( deltaZ > 2.*zmax || deltaZ < 2.*zmin) success = false;
  }

  return success;
      
}

float GSPixelHitMatcher::getVertex(){

  return vertex;
}






