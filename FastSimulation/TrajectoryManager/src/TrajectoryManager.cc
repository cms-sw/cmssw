//CMSSW Headers
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/Surface.h"
#include "Geometry/Surface/interface/TangentPlane.h"
//#include "CommonReco/PatternTools/interface/DummyDet.h"

// Tracker reco geometry headers 
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FastSimulation/TrajectoryManager/interface/InsideBoundsMeasurementEstimator.h"
//#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "MagneticField/UniformEngine/src/UniformMagneticField.h"

//FAMOS Headers
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include "FastSimulation/ParticleDecay/interface/Pythia6Decays.h"
//#include "FastSimulation/FamosTracker/interface/FamosDummyDet.h"
//#include "FastSimulation/FamosTracker/interface/FamosBasicRecHit.h"

#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Event/interface/KineParticleFilter.h"

#include "FastSimulation/Utilities/interface/Histos.h"
//#include "FastSimulation/Utilities/interface/FamosLooses.h"

#include <iostream>
#include <list>
#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

TrajectoryManager::TrajectoryManager(FSimEvent* aSimEvent, 
				     const edm::ParameterSet& matEff,
				     const edm::ParameterSet& simHits,
				     bool activateDecays) : 
  mySimEvent(aSimEvent), 
  _theGeometry(0), 
  theMaterialEffects(0), 
  myDecayEngine(0), 
  theGeomTracker(0),
  theGeomSearchTracker(0),
  theLayerMap(56, static_cast<const DetLayer*>(0)), // reserve space for layers here
  theNegLayerOffset(27),
  myHistos(0)

{
  
  // Initialize Bthe stable particle decay engine 
  if ( activateDecays ) myDecayEngine = new Pythia6Decays();

  // Initialize the Material Effects updator, if needed
  if ( matEff.getParameter<bool>("PairProduction") || 
       matEff.getParameter<bool>("Bremsstrahlung") ||
       matEff.getParameter<bool>("EnergyLoss") || 
       matEff.getParameter<bool>("MultipleScattering") )
       theMaterialEffects = new MaterialEffects(matEff);

  // Save SimHits according to Optiom
  // Only the hits from first half loop is saved
  firstLoop = simHits.getUntrackedParameter<bool>("firstLoop",true);
  // Only if pT>pTmin are the hits saved
  pTmin = simHits.getUntrackedParameter<double>("pTmin",0.5);

  thePSimHits = new vector<PSimHit>();
  thePSimHits->reserve(200000);

  // Get the Famos Histos pointer
  myHistos = Histos::instance();

  // Initialize a few histograms
  /* 
  myHistos->book("h300",1210,-121.,121.,1210,-121.,121.);
  myHistos->book("h301",1200,-300.,300.,1210,-121.,121.);
  */
}

void 
TrajectoryManager::initializeRecoGeometry(const GeometricSearchTracker* geomSearchTracker) { 
  
  theGeomSearchTracker = geomSearchTracker;

  // Initialize the simplified tracker geometry
  _theGeometry = new TrackerInteractionGeometry(theGeomSearchTracker);

  initializeLayerMap();

}

void 
TrajectoryManager::initializeTrackerGeometry(const TrackerGeometry* geomTracker) { 
  
  theGeomTracker = geomTracker;

}

TrackerInteractionGeometry*
TrajectoryManager::theGeometry() {
  return _theGeometry;
}

TrajectoryManager::~TrajectoryManager() {
  if ( _theGeometry ) delete _theGeometry;
  if ( myDecayEngine ) delete myDecayEngine;
  if ( theMaterialEffects ) delete theMaterialEffects;
  if ( thePSimHits ) delete thePSimHits;
  //Write the histograms
  //myHistos->put("histos.root");
  if ( myHistos ) delete myHistos;

}

void
TrajectoryManager::reconstruct()
{

  // Clear the hits of the previous event
  thePSimHits->clear();

  // The new event
  HepLorentzVector myBeamPipe = HepLorentzVector(0.,25., 9999999.,0.);

  std::list<TrackerLayer>::iterator cyliter;

  ParticlePropagator P_before;

  //  bool debug = mySimEvent->id().event() == 62;

  // Loop over the particles
  for( int fsimi=0; fsimi < (int) mySimEvent->nTracks() ; ++fsimi) {

    FSimTrack& myTrack = mySimEvent->track(fsimi);

    // If the particle has decayed inside the beampipe, or decays 
    // immediately, there is nothing to do
    if( !myTrack.notYetToEndVertex(myBeamPipe) ) continue;
    myTrack.setPropagate();

    // Get the geometry elements 
    cyliter = _theGeometry->cylinderBegin();
    // Prepare the propagation  
    ParticlePropagator PP(myTrack);

    //The real work starts here
    int success = 1;
    int sign = +1;
    int loop = 0;
    int cyl = 0;

    // Find the initial cylinder to propagate to.      
    for ( ; cyliter != _theGeometry->cylinderEnd() ; ++cyliter ) {
      
      PP.setPropagationConditions(*cyliter);
      if ( PP.inside() && !PP.onSurface() ) break;
      ++cyl;

    }

    // Loop over the cylinders
    while ( cyliter != _theGeometry->cylinderEnd() &&
	    loop<100 &&                            // No more than 100 loops
	    myTrack.notYetToEndVertex(PP.vertex())) { // The particle decayed

      // To prevent from interacting twice in a row with the same layer
      bool escapeBarrel    = (PP.getSuccess() == -1 && success == 1);
      bool escapeEndcap    = (PP.getSuccess() == -2 && success == 1);
      // To break the loop
      bool fullPropagation = 
	(PP.getSuccess() <= 0 && success==0) || escapeEndcap;

      // Define the propagation conditions
      if ( escapeBarrel ) { 
	if ( ++cyliter ==_theGeometry->cylinderEnd() ) {
	  --cyliter; fullPropagation=true; 
	} else {
	  sign=1; ++cyl; 
	}
      }

      PP.setPropagationConditions(*cyliter,!fullPropagation);
      if ( escapeEndcap ) PP.increaseRCyl(0.0005);

      // Remember last propagation outcome
      success = PP.getSuccess();
	  
      // Propagation was not successful :
      // Change the sign of the cylinder increment and count the loops
      if ( !PP.propagateToBoundSurface(*cyliter) || 
	   PP.getSuccess()<=0) {
	sign = -sign;
	++loop;
      }

      // The particle may have decayed on its way... in which the daughters
      // have to be added to the event record
      if ( PP.hasDecayed() ) updateWithDaughters(PP,fsimi);
      if ( PP.hasDecayed() ) break;

      // Exit by the endcaps or innermost cylinder :
      // Positive cylinder increment
      if ( PP.getSuccess()==2 || cyliter==_theGeometry->cylinderBegin() ) 
	sign = +1; 
	  
      // Successful propagation to a cylinder, with some Material :
      if( PP.getSuccess() > 0 && PP.onFiducial() ) {

	bool saveHit = 
	  ( (loop==0 && sign>0) || !firstLoop ) &&   // Save only first half loop
	  PP.charge()!=0. &&                         // Consider only charged particles
	  cyliter->sensitive() &&                    // Consider only sensitive layers
	  PP.perp()>pTmin;                           // Consider only pT > pTmin

	// Save Particle before Material Effects
	if ( saveHit ) P_before = ParticlePropagator(PP); 

	// Material effects are simulated there
	if ( theMaterialEffects ) 
	  theMaterialEffects->interact(*mySimEvent,*cyliter,PP,fsimi); 

	if ( saveHit ) { 

	  // Consider only active layers
	  if ( cyliter->layerNumber() ) {
	    // Add information to the FSimTrack (not yet available)
	    //	    myTrack.addSimHit(PP,layer);

	    // Return one or two (for overlap regions) PSimHits in the full 
	    // tracker geometry
	    if ( theGeomTracker ) 
	      createPSimHits(*cyliter, P_before, PP, fsimi,myTrack.type());

	  }
	}

	// Fill Histos (~poor man event display)
	/* 
	myHistos->fill("h300",PP.x(),PP.y());
	if ( sin(PP.vertex().phi()) > 0. ) 
	  myHistos->fill("h301",PP.z(),PP.vertex().perp());
	else
	  myHistos->fill("h301",PP.z(),-PP.vertex().perp());
	*/

	//The particle may have lost its energy in the material
	if ( myTrack.notYetToEndVertex(PP.vertex()) && 
	     !mySimEvent->filter().accept(PP)  ) 
	  mySimEvent->addSimVertex(PP.vertex(),fsimi);
	  
      }

      // Stop here if the particle has reached an end
      if ( myTrack.notYetToEndVertex(PP.vertex()) ) {

	// Otherwise increment the cylinder iterator
	//	do { 
	if (sign==1) {++cyliter;++cyl;}
	else         {--cyliter;--cyl;}

	// Check if the last surface has been reached 
	if( cyliter==_theGeometry->cylinderEnd()) {

	  // Try to propagate to the ECAL in half a loop
	  // Note: Layer1 = ECAL Barrel entrance, or Preshower
	  // entrance, or ECAL Endcap entrance (in the corner)
	  PP.propagateToEcal();
	  // PP.propagateToPreshowerLayer1();

	  // If it is not possible, try go back to the last cylinder
	  if(PP.getSuccess()==0) {
	    --cyliter; --cyl; sign = -sign;
	    PP.setPropagationConditions(*cyliter);
	    PP.propagateToBoundSurface(*cyliter);

	    // If there is definitely no way, leave it here.
	    if(PP.getSuccess()<0) {++cyliter; ++cyl;}

	  }

	  // Check if the particle has decayed on the way to ECAL
	  if ( PP.hasDecayed() )
	    updateWithDaughters(PP,fsimi);

	}
      }

    }

    // Propagate all particles without a end vertex to the Preshower, 
    // theECAL and the HCAL.
    if ( myTrack.notYetToEndVertex(PP.vertex()) )
      propagateToCalorimeters(PP,fsimi);

  }

}

void 
TrajectoryManager::propagateToCalorimeters(ParticlePropagator& PP, int fsimi) {

  FSimTrack& myTrack = mySimEvent->track(fsimi);

  // Propagate to Preshower Layer 1 
  PP.propagateToPreshowerLayer1(false);
  if ( PP.hasDecayed() ) {
    updateWithDaughters(PP,fsimi);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) && PP.getSuccess() > 0 )
    myTrack.setLayer1(PP,PP.getSuccess());
  
  // Propagate to Preshower Layer 2 
  PP.propagateToPreshowerLayer2(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP,fsimi);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) && PP.getSuccess() > 0 )
    myTrack.setLayer2(PP,PP.getSuccess());

  // Propagate to Ecal Endcap
  PP.propagateToEcalEntrance(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP,fsimi);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) )
    myTrack.setEcal(PP,PP.getSuccess());

  // Propagate to HCAL entrance
  PP.propagateToHcalEntrance(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP,fsimi);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) )
    myTrack.setHcal(PP,PP.getSuccess());

  // Propagate to VFCAL entrance
  PP.propagateToVFcalEntrance(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP,fsimi);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) )
    myTrack.setVFcal(PP,PP.getSuccess());
    
}

bool
TrajectoryManager::propagateToLayer(ParticlePropagator& PP, unsigned layer) {

  std::list<TrackerLayer>::iterator cyliter;
  bool done = false;

  // Get the geometry elements 
  cyliter = _theGeometry->cylinderBegin();

  // Find the layer to propagate to.      
  for ( ; cyliter != _theGeometry->cylinderEnd() ; ++cyliter ) {

    if ( layer != cyliter->layerNumber() ) continue;
      
    PP.setPropagationConditions(*cyliter);

    done =  
      PP.propagateToBoundSurface(*cyliter) &&
      PP.getSuccess() > 0 && 
      PP.onFiducial();

    break;
    
  }

  return done;

}

void
TrajectoryManager::updateWithDaughters(ParticlePropagator& PP, int fsimi) {

  // Decays are not activated : do nothing
  if ( !myDecayEngine ) return;

  // Invoke PYDECY to decay the particle and get the daughters
  DaughterParticleList daughters = myDecayEngine->particleDaughters(PP);
  
  // Update the FSimEvent with an end vertex and with the daughters
  if ( daughters.size() ) { 
    DaughterParticleIterator daughter = daughters.begin();
    
    int ivertex = mySimEvent->addSimVertex((*daughter)->vertex(),fsimi);

    if ( ivertex != -1 ) {
      for ( ; daughter != daughters.end(); ++daughter) 
	mySimEvent->addSimTrack(*daughter, ivertex);
    }
  }
}


void
TrajectoryManager::createPSimHits(const TrackerLayer& layer,
				  ParticlePropagator& P_before,
				  ParticlePropagator& P_after,
				  int trackID, int partID) {

  //  float eloss = (P_before.momentum().e()-P_after.momentum().e());
  float eloss = (P_before.e()-P_after.e());

  // Propagate the particle coordinates to the closest tracker detector(s) 
  // in this layer and create the PSimHit(s)

  //  const MagneticField& mf = MagneticFieldMap::instance()->magneticField();
  // This solution is actually much faster !
  UniformMagneticField mf(P_before.getMagneticField());
  AnalyticalPropagator alongProp(&mf, anyDirection);
  InsideBoundsMeasurementEstimator est;

  typedef GeometricSearchDet::DetWithState   DetWithState;
  const DetLayer* tkLayer = detLayer(layer,P_before.z());
  const ParticlePropagator& cpp(P_before);
  TrajectoryStateOnSurface trajState = makeTrajectoryState( tkLayer, cpp, &mf);

  // Find, in the corresponding layers, the detectors compatible 
  // with the current track 
  std::vector<DetWithState> compat 
    = tkLayer->compatibleDets( trajState, alongProp, est);

  // And create the corresponding PSimHits
  for (std::vector<DetWithState>::const_iterator i=compat.begin(); i!=compat.end(); i++) {
    makePSimHits( i->first, i->second, *thePSimHits, trackID, eloss, partID);
  }

}

TrajectoryStateOnSurface 
TrajectoryManager::makeTrajectoryState( const DetLayer* layer, 
					const ParticlePropagator& pp,
					const MagneticField* field) const
{
  GlobalPoint  pos( pp.x(), pp.y(), pp.z());
  GlobalVector mom( pp.px(), pp.py(), pp.pz());
  ReferenceCountingPointer<TangentPlane> plane = layer->surface().tangentPlane(pos);
  return TrajectoryStateOnSurface
    (GlobalTrajectoryParameters( pos, mom, TrackCharge( pp.charge()), field), *plane);
}

void 
TrajectoryManager::makePSimHits( const GeomDet* det, 
				 const TrajectoryStateOnSurface& ts,
				 std::vector<PSimHit>& result, int tkID, 
				 float el, int pID) const
{
  std::vector< const GeomDet*> comp = det->components();
  if (!comp.empty()) {
    for (std::vector< const GeomDet*>::const_iterator i = comp.begin();
	 i != comp.end(); i++) {
      const GeomDetUnit* du = dynamic_cast<const GeomDetUnit*>(*i);
      if (du != 0) result.push_back( makeSinglePSimHit( *du, ts, tkID, el, pID));
    }
  }
  else {
    const GeomDetUnit* du = dynamic_cast<const GeomDetUnit*>(det);
    if (du != 0) result.push_back( makeSinglePSimHit( *du, ts, tkID, el, pID));
  }
}

PSimHit 
TrajectoryManager::makeSinglePSimHit( const GeomDetUnit& det,
				      const TrajectoryStateOnSurface& ts, 
				      int tkID, float el, int pID) const
{
  const float onSurfaceTolarance = 0.01; // 10 microns

  LocalPoint lpos;
  LocalVector lmom;
  if ( fabs( det.toLocal(ts.globalPosition()).z()) < onSurfaceTolarance) {
    lpos = ts.localPosition();
    lmom = ts.localMomentum();
  }
  else {
    HelixArbitraryPlaneCrossing crossing( ts.globalPosition().basicVector(),
					  ts.globalMomentum().basicVector(),
					  ts.transverseCurvature(),
					  anyDirection);
    std::pair<bool,double> path = crossing.pathLength(det.surface());
    if (!path.first) {
      edm::LogError("FastTracker") << "TrajectoryManager ERROR: crossing with det failed, skipping PSimHit";
      return  PSimHit();
    }
    lpos = det.toLocal( GlobalPoint( crossing.position(path.second)));
    lmom = det.toLocal( GlobalVector( crossing.direction(path.second)));
    lmom = lmom.unit() * ts.localMomentum().mag();
  }

  float halfThick = 0.5*det.surface().bounds().thickness();
  float pZ = lmom.z();
  LocalPoint entry = lpos + (-halfThick/pZ) * lmom;
  LocalPoint exit = lpos + halfThick/pZ * lmom;
  float tof = ts.globalPosition().mag() / 30. ; // in nanoseconds, FIXME: very approximate
  float eloss = el; // FIXME should be dependent on thickness and crossing angle...

  // FIXME: fix the track ID and the particle ID
  PSimHit hit( entry, exit, lmom.mag(), tof, eloss, pID,
		  det.geographicalId().rawId(), tkID,
		  lmom.theta(),
		  lmom.phi());
  // Fill Histos (~poor man event display)
  /* 
  GlobalPoint gpos( det.toGlobal(hit.localPosition()));
  myHistos->fill("h300",gpos.x(),gpos.y());
  if ( sin(gpos.phi()) > 0. ) 
    myHistos->fill("h301",gpos.z(),gpos.perp());
  else
    myHistos->fill("h301",gpos.z(),-gpos.perp());
  */

  return hit;
}

void 
TrajectoryManager::initializeLayerMap()
{

// These are the BoundSurface&, the BoundDisk* and the BoundCylinder* for that layer
//   const BoundSurface& theSurface = layer.surface();
//   BoundDisk* theDisk = layer.disk();  // non zero for endcaps
//   BoundCylinder* theCylinder = layer.cylinder(); // non zero for barrel
//   int theLayer = layer.layerNumber(); // 1->3 PixB, 4->5 PixD, 
//                                       // 6->9 TIB, 10->12 TID, 
//                                       // 13->18 TOB, 19->27 TEC

/// ATTENTION: HARD CODED LOGIC! If Famos layer numbering changes this logic needs to 
/// be adapted to the new numbering!

  std::vector< BarrelDetLayer*>   barrelLayers = 
    theGeomSearchTracker->barrelLayers();
  LogDebug("FastTracker") << "Barrel DetLayer dump: ";
  for (std::vector< BarrelDetLayer*>::const_iterator bl=barrelLayers.begin();
       bl != barrelLayers.end(); ++bl) {
    LogDebug("FastTracker")<< "radius " << (**bl).specificSurface().radius(); 
  }

  std::vector< ForwardDetLayer*>  posForwardLayers = 
    theGeomSearchTracker->posForwardLayers();
  LogDebug("FastTracker") << "Positive Forward DetLayer dump: ";
  for (std::vector< ForwardDetLayer*>::const_iterator fl=posForwardLayers.begin();
       fl != posForwardLayers.end(); ++fl) {
    LogDebug("FastTracker") << "Z pos "
			    << (**fl).surface().position().z()
			    << " radii " 
			    << (**fl).specificSurface().innerRadius() 
			    << ", " 
			    << (**fl).specificSurface().outerRadius(); 
  }

  const float rTolerance = 1.5;
  const float zTolerance = 3.;

  LogDebug("FastTracker")<< "Dump of TrackerInteractionGeometry cylinders:";
  for( std::list<TrackerLayer>::iterator i=_theGeometry->cylinderBegin();
       i!=_theGeometry->cylinderEnd(); ++i) {
    const BoundCylinder* cyl = i->cylinder();
    const BoundDisk* disk = i->disk();

    LogDebug("FastTracker") << "Famos Layer no " << i->layerNumber()
			    << " is sensitive? " << i->sensitive()
			    << " pos " << i->surface().position();
    if (!i->sensitive()) continue;

    if (cyl != 0) {
      LogDebug("FastTracker") << " cylinder radius " << cyl->radius();
      bool found = false;
      for (std::vector< BarrelDetLayer*>::const_iterator 
	     bl=barrelLayers.begin(); bl != barrelLayers.end(); ++bl) {
	if (fabs( cyl->radius() - (**bl).specificSurface().radius()) < rTolerance) {
	  theLayerMap[i->layerNumber()] = *bl;
	  found = true;
	  LogDebug("FastTracker")<< "Corresponding DetLayer found with radius "
				 << (**bl).specificSurface().radius();
	  break;
	}
      }
      if (!found) {
	edm::LogError("FastTracker") << "FAILED to find a corresponding DetLayer!";
      }
    }
    else {
      LogDebug("FastTracker") << " disk radii " << disk->innerRadius() 
		 << ", " << disk->outerRadius();
      bool found = false;
      for (std::vector< ForwardDetLayer*>::const_iterator fl=posForwardLayers.begin();
	   fl != posForwardLayers.end(); ++fl) {
	if (fabs( disk->position().z() - (**fl).surface().position().z()) < zTolerance) {
	  theLayerMap[i->layerNumber()] = *fl;
	  found = true;
	  LogDebug("FastTracker") << "Corresponding DetLayer found with Z pos "
				  << (**fl).surface().position().z()
				  << " and radii " 
				  << (**fl).specificSurface().innerRadius() 
				  << ", " 
				  << (**fl).specificSurface().outerRadius(); 
	  break;
	}
      }
      if (!found) {
	edm::LogError("FastTracker") << "FAILED to find a corresponding DetLayer!";
      }
    }
  }

  // Put the negative layers in the same map but with an offset
  std::vector< ForwardDetLayer*>  negForwardLayers = theGeomSearchTracker->negForwardLayers();
  for (std::vector< ForwardDetLayer*>::const_iterator nl=negForwardLayers.begin();
       nl != negForwardLayers.end(); ++nl) {
    for (int i=0; i<=theNegLayerOffset; i++) {
      if (theLayerMap[i] == 0) continue;
      if ( fabs( (**nl).surface().position().z() +theLayerMap[i]-> surface().position().z()) < zTolerance) {
	theLayerMap[i+theNegLayerOffset] = *nl;
	break;
      }
    }
  }  

}

const DetLayer*  
TrajectoryManager::detLayer( const TrackerLayer& layer, float zpos) const
{
  if (zpos > 0 || !layer.forward() ) return theLayerMap[layer.layerNumber()];
  else return theLayerMap[layer.layerNumber()+theNegLayerOffset];
}

void 
TrajectoryManager::loadSimHits(edm::PSimHitContainer & c) const
{
  for(std::vector<PSimHit>::const_iterator 
	it = thePSimHits->begin();
        it!= thePSimHits->end();it++) c.push_back(*it);
}
