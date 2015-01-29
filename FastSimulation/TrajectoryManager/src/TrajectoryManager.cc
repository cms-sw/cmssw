//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//CMSSW Headers
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"

// Tracker reco geometry headers 
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FastSimulation/TrajectoryManager/interface/InsideBoundsMeasurementEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//FAMOS Headers
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"
#include "FastSimulation/TrajectoryManager/interface/LocalMagneticField.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include "FastSimulation/ParticleDecay/interface/PythiaDecays.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

//#include "FastSimulation/Utilities/interface/Histos.h"
//#include "FastSimulation/Utilities/interface/FamosLooses.h"
// Numbering scheme

//#define FAMOS_DEBUG
#ifdef FAMOS_DEBUG
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#endif

#include <list>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

TrajectoryManager::TrajectoryManager(FSimEvent* aSimEvent, 
				     const edm::ParameterSet& matEff,
				     const edm::ParameterSet& simHits,
				     const edm::ParameterSet& decays) :
  mySimEvent(aSimEvent), 
  _theGeometry(0),
  _theFieldMap(0),
  theMaterialEffects(0), 
  myDecayEngine(0), 
  theGeomTracker(0),
  theGeomSearchTracker(0),
  theLayerMap(56, static_cast<const DetLayer*>(0)), // reserve space for layers here
  theNegLayerOffset(27),
  //  myHistos(0),
  use_hardcoded(1)

{  
  //std::cout << "TrajectoryManager.cc 1 use_hardcoded = " << use_hardcoded << std::endl;
  use_hardcoded = matEff.getParameter<bool>("use_hardcoded_geometry");

  // Initialize Bthe stable particle decay engine 
  if ( decays.getParameter<bool>("ActivateDecays")) { 
    myDecayEngine = new PythiaDecays();
    distCut = decays.getParameter<double>("DistCut");
  }
  // Initialize the Material Effects updator, if needed
  if ( matEff.getParameter<bool>("PairProduction") || 
       matEff.getParameter<bool>("Bremsstrahlung") ||
       matEff.getParameter<bool>("MuonBremsstrahlung") ||
       matEff.getParameter<bool>("EnergyLoss") || 
       matEff.getParameter<bool>("MultipleScattering") || 
       matEff.getParameter<bool>("NuclearInteraction")
       )
       theMaterialEffects = new MaterialEffects(matEff);

  // Save SimHits according to Optiom
  // Only the hits from first half loop is saved
  firstLoop = simHits.getUntrackedParameter<bool>("firstLoop",true);
  // Only if pT>pTmin are the hits saved
  pTmin = simHits.getUntrackedParameter<double>("pTmin",0.5);

  /*
  // Get the Famos Histos pointer
  myHistos = Histos::instance();

  // Initialize a few histograms
   
  myHistos->book("h302",1210,-121.,121.,1210,-121.,121.);
  myHistos->book("h300",1210,-121.,121.,1210,-121.,121.);
  myHistos->book("h301",1200,-300.,300.,1210,-121.,121.);  
  myHistos->book("h303",1200,-300.,300.,1210,-121.,121.);
  */
}

void 
TrajectoryManager::initializeRecoGeometry(const GeometricSearchTracker* geomSearchTracker,
					  const TrackerInteractionGeometry* interactionGeometry,
					  const MagneticFieldMap* aFieldMap)
{
  
  // Initialize the reco tracker geometry
  theGeomSearchTracker = geomSearchTracker;
  
  // Initialize the simplified tracker geometry
  _theGeometry = interactionGeometry;

  initializeLayerMap();

  // Initialize the magnetic field
  _theFieldMap = aFieldMap;

}

void 
TrajectoryManager::initializeTrackerGeometry(const TrackerGeometry* geomTracker) { 
  
  theGeomTracker = geomTracker;

}

const TrackerInteractionGeometry*
TrajectoryManager::theGeometry() {
  return _theGeometry;
}

TrajectoryManager::~TrajectoryManager() {

  if ( myDecayEngine ) delete myDecayEngine;
  if ( theMaterialEffects ) delete theMaterialEffects;

  //Write the histograms
  /*
  myHistos->put("histos.root");
  if ( myHistos ) delete myHistos;
  */
}

void
TrajectoryManager::reconstruct(const TrackerTopology *tTopo, RandomEngineAndDistribution const* random)
{

  // Clear the hits of the previous event
  //  thePSimHits->clear();
  thePSimHits.clear();

  // The new event
  XYZTLorentzVector myBeamPipe = XYZTLorentzVector(0.,2.5, 9999999.,0.);

  std::list<TrackerLayer>::const_iterator cyliter;

  // bool debug = mySimEvent->id().event() == 8;

  // Loop over the particles (watch out: increasing upper limit!)
  for( int fsimi=0; fsimi < (int) mySimEvent->nTracks(); ++fsimi) {
    // If the particle has decayed inside the beampipe, or decays 
    // immediately, there is nothing to do
    //if ( debug ) std::cout << mySimEvent->track(fsimi) << std::endl;
    //if ( debug ) std::cout << "Not yet at end vertex ? " << mySimEvent->track(fsimi).notYetToEndVertex(myBeamPipe) << std::endl;
    if( !mySimEvent->track(fsimi).notYetToEndVertex(myBeamPipe) ) continue;
    mySimEvent->track(fsimi).setPropagate();

    // Get the geometry elements 
    cyliter = _theGeometry->cylinderBegin();
    // Prepare the propagation  
    ParticlePropagator PP(mySimEvent->track(fsimi),_theFieldMap,random);
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

    // The particle has a pseudo-rapidity (position or momentum direction) 
    // in excess of 3.0. Just simply go to the last tracker layer
    // without bothering with all the details of the propagation and 
    // material effects.
    // 08/02/06 - pv: increase protection from 0.99 (eta=2.9932) to 0.9998 (eta=4.9517)
    //                to simulate material effects at large eta 
    // if above 0.99: propagate to the last tracker cylinder where the material is concentrated!
    double ppcos2T =  PP.cos2Theta();
    double ppcos2V =  PP.cos2ThetaV();

    if(use_hardcoded){
      if ( ( ppcos2T > 0.99 && ppcos2T < 0.9998 ) && ( cyl == 0 || ( ppcos2V > 0.99 && ppcos2V < 0.9998 ) ) ){ 
	if ( cyliter != _theGeometry->cylinderEnd() ) { 
	  cyliter = _theGeometry->cylinderEnd(); 
	  --cyliter;
	}
	// if above 0.9998: don't propagate at all (only to the calorimeters directly)
      } else if ( ppcos2T > 0.9998 && ( cyl == 0 || ppcos2V > 0.9998 ) ) { 
	cyliter = _theGeometry->cylinderEnd();
      } 
    }
    else {
      if ( ppcos2T > 0.9998 && ( cyl == 0 || ppcos2V > 0.9998 ) ) { 
	cyliter = _theGeometry->cylinderEnd();
      }
    }
	
    // Loop over the cylinders
    while ( cyliter != _theGeometry->cylinderEnd() &&
	    loop<100 &&                            // No more than 100 loops
	    mySimEvent->track(fsimi).notYetToEndVertex(PP.vertex())) { // The particle decayed

      // Skip layers with no material (kept just for historical reasons)
      if ( cyliter->surface().mediumProperties().radLen() < 1E-10 ) { 
	++cyliter; ++cyl;
	continue;
      }
      
      // Pathological cases:
      // To prevent from interacting twice in a row with the same layer
      //      bool escapeBarrel    = (PP.getSuccess() == -1 && success == 1);
      bool escapeBarrel    = PP.getSuccess() == -1;
      bool escapeEndcap    = (PP.getSuccess() == -2 && success == 1);
      // To break the loop
      bool fullPropagation = 
	(PP.getSuccess() <= 0 && success==0) || escapeEndcap;

      if ( escapeBarrel ) {
	++cyliter; ++cyl;
	while (cyliter != _theGeometry->cylinderEnd() && cyliter->forward() ) {
	  sign=1; ++cyliter; ++cyl;
	}

	if ( cyliter == _theGeometry->cylinderEnd()  ) {
	  --cyliter; --cyl; fullPropagation=true; 
	}

      }

      // Define the propagation conditions
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
      if ( PP.hasDecayed() || (!mySimEvent->track(fsimi).nDaughters() && PP.PDGcTau()<1E-3 ) ) { 
	updateWithDaughters(PP, fsimi, random);
	break;
      }

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
	  PP.Perp2()>pTmin*pTmin;                    // Consider only pT > pTmin

        // Material effects are simulated there
	if ( theMaterialEffects )
          theMaterialEffects->interact(*mySimEvent,*cyliter,PP,fsimi, random);

	// There is a PP.setXYZT=(0,0,0,0) if bremss fails
	saveHit &= PP.E()>1E-6;

	if ( saveHit ) { 
	  // Consider only active layers
	  if ( cyliter->sensitive() ) {
	    // Add information to the FSimTrack (not yet available)
	    //	    myTrack.addSimHit(PP,layer);

	    // Return one or two (for overlap regions) PSimHits in the full 
	    // tracker geometry
	    if ( theGeomTracker ) 
	      createPSimHits(*cyliter, PP, thePSimHits[fsimi], fsimi,mySimEvent->track(fsimi).type(), tTopo);

	    /*
	    myHistos->fill("h302",PP.X() ,PP.Y());
	    if ( sin(PP.vertex().Phi()) > 0. ) 
	      myHistos->fill("h303",PP.Z(),PP.R());
	    else
	      myHistos->fill("h303",PP.Z(),-PP.R());
	    */

	  }
	}

	// Fill Histos (~poor man event display)
	/*	 
	myHistos->fill("h300",PP.x(),PP.y());
	if ( sin(PP.vertex().phi()) > 0. ) 
	  myHistos->fill("h301",PP.z(),sqrt(PP.vertex().Perp2()));
	else
	  myHistos->fill("h301",PP.z(),-sqrt(PP.vertex().Perp2()));
	*/

	//The particle may have lost its energy in the material
	if ( mySimEvent->track(fsimi).notYetToEndVertex(PP.vertex()) && 
	     !mySimEvent->filter().accept(PP)  ) 
	  mySimEvent->addSimVertex(PP.vertex(),fsimi, FSimVertexType::END_VERTEX);
	  
      }

      // Stop here if the particle has reached an end
      if ( mySimEvent->track(fsimi).notYetToEndVertex(PP.vertex()) ) {

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
	    updateWithDaughters(PP, fsimi, random);

	}
      }

    }

    // Propagate all particles without a end vertex to the Preshower, 
    // theECAL and the HCAL.
    if ( mySimEvent->track(fsimi).notYetToEndVertex(PP.vertex()) )
      propagateToCalorimeters(PP, fsimi, random);

  }

  // Save the information from Nuclear Interaction Generation
  if ( theMaterialEffects ) theMaterialEffects->save();

}

void 
TrajectoryManager::propagateToCalorimeters(ParticlePropagator& PP, int fsimi, RandomEngineAndDistribution const* random) {

  FSimTrack& myTrack = mySimEvent->track(fsimi);

  // Set the position and momentum at the end of the tracker volume
  myTrack.setTkPosition(PP.vertex().Vect());
  myTrack.setTkMomentum(PP.momentum());

  // Propagate to Preshower Layer 1 
  PP.propagateToPreshowerLayer1(false);
  if ( PP.hasDecayed() ) {
    updateWithDaughters(PP, fsimi, random);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) && PP.getSuccess() > 0 )
    myTrack.setLayer1(PP,PP.getSuccess());
  
  // Propagate to Preshower Layer 2 
  PP.propagateToPreshowerLayer2(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP, fsimi, random);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) && PP.getSuccess() > 0 )
    myTrack.setLayer2(PP,PP.getSuccess());

  // Propagate to Ecal Endcap
  PP.propagateToEcalEntrance(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP, fsimi, random);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) )
    myTrack.setEcal(PP,PP.getSuccess());

  // Propagate to HCAL entrance
  PP.propagateToHcalEntrance(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP,fsimi, random);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) )
    myTrack.setHcal(PP,PP.getSuccess());

  // Propagate to VFCAL entrance
  PP.propagateToVFcalEntrance(false);
  if ( PP.hasDecayed() ) { 
    updateWithDaughters(PP,fsimi, random);
    return;
  }
  if ( myTrack.notYetToEndVertex(PP.vertex()) )
    myTrack.setVFcal(PP,PP.getSuccess());
    
}

bool
TrajectoryManager::propagateToLayer(ParticlePropagator& PP, unsigned layer) {

  std::list<TrackerLayer>::const_iterator cyliter;
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
TrajectoryManager::updateWithDaughters(ParticlePropagator& PP, int fsimi, RandomEngineAndDistribution const* random) {

  // The particle was already decayed in the GenEvent, but still the particle was 
  // allowed to propagate (for magnetic field bending, for material effects, etc...)
  // Just modify the momentum of the daughters in that case 
  unsigned nDaugh = mySimEvent->track(fsimi).nDaughters();
  if ( nDaugh ) {

    // Move the vertex
    unsigned vertexId = mySimEvent->track(fsimi).endVertex().id();
    mySimEvent->vertex(vertexId).setPosition(PP.vertex());

    // Before-propagation and after-propagation momentum and vertex position
    XYZTLorentzVector momentumBefore = mySimEvent->track(fsimi).momentum();
    XYZTLorentzVector momentumAfter = PP.momentum();
    double magBefore = std::sqrt(momentumBefore.Vect().mag2());
    double magAfter = std::sqrt(momentumAfter.Vect().mag2());
    // Rotation to be applied
    XYZVector axis = momentumBefore.Vect().Cross(momentumAfter.Vect());
    double angle = std::acos(momentumBefore.Vect().Dot(momentumAfter.Vect())/(magAfter*magBefore));
    Rotation r(axis,angle);
    // Rescaling to be applied
    double rescale = magAfter/magBefore;

    // Move, rescale and rotate daugthers, grand-daughters, etc. 
    moveAllDaughters(fsimi,r,rescale);

  // The particle is not decayed in the GenEvent, decay it with PYTHIA 
  } else { 

    // Decays are not activated : do nothing
    if ( !myDecayEngine ) return;

    // Invoke PYDECY (Pythia6) or Pythia8 to decay the particle and get the daughters
    const DaughterParticleList& daughters =  myDecayEngine->particleDaughters(PP, &random->theEngine());

    // Update the FSimEvent with an end vertex and with the daughters
    if ( daughters.size() ) { 
      double distMin = 1E99;
      int theClosestChargedDaughterId = -1;
      DaughterParticleIterator daughter = daughters.begin();
      
      int ivertex = mySimEvent->addSimVertex(daughter->vertex(),fsimi, 
					     FSimVertexType::DECAY_VERTEX);
      
      if ( ivertex != -1 ) {
	for ( ; daughter != daughters.end(); ++daughter) {
	  int theDaughterId = mySimEvent->addSimTrack(&(*daughter), ivertex);
	  // Find the closest charged daughter (if charged mother)
	  if ( PP.charge() * daughter->charge() > 1E-10 ) {
	    double dist = (daughter->Vect().Unit().Cross(PP.Vect().Unit())).R();
	    if ( dist < distCut && dist < distMin ) { 
	      distMin = dist;
	      theClosestChargedDaughterId = theDaughterId;
	    }
	  }
	}
      }
      // Attach mother and closest daughter sp as to cheat tracking ;-)
      if ( theClosestChargedDaughterId >=0 ) 
	mySimEvent->track(fsimi).setClosestDaughterId(theClosestChargedDaughterId);
    }

  }

}


void
TrajectoryManager::moveAllDaughters(int fsimi, const Rotation& r, double rescale) { 

  //
  for ( unsigned idaugh=0; idaugh < (unsigned)(mySimEvent->track(fsimi).nDaughters()); ++idaugh) { 
    // Initial momentum of the daughter
    XYZTLorentzVector daughMomentum (mySimEvent->track(fsimi).daughter(idaugh).momentum()); 
    // Rotate and rescale
    XYZVector newMomentum (r * daughMomentum.Vect()); 
    newMomentum *= rescale;
    double newEnergy = std::sqrt(newMomentum.mag2() + daughMomentum.mag2());
    // Set the new momentum
    mySimEvent->track(fsimi).setMomentum(XYZTLorentzVector(newMomentum.X(),newMomentum.Y(),newMomentum.Z(),newEnergy));
    // Watch out : recursive call to get all grand-daughters
    int fsimDaug = mySimEvent->track(fsimi).daughter(idaugh).id();
    moveAllDaughters(fsimDaug,r,rescale);
  }
}

void
TrajectoryManager::createPSimHits(const TrackerLayer& layer,
                                  const ParticlePropagator& PP,
				  std::map<double,PSimHit>& theHitMap,
				  int trackID, int partID, const TrackerTopology *tTopo) {

  // Propagate the particle coordinates to the closest tracker detector(s) 
  // in this layer and create the PSimHit(s)

  //  const MagneticField& mf = MagneticFieldMap::instance()->magneticField();
  // This solution is actually much faster !
  LocalMagneticField mf(PP.getMagneticField());
  AnalyticalPropagator alongProp(&mf, anyDirection);
  InsideBoundsMeasurementEstimator est;

//   std::cout << "PP.X() = " << PP.X() << std::endl;
//   std::cout << "PP.Y() = " << PP.Y() << std::endl;
//   std::cout << "PP.Z() = " << PP.Z() << std::endl;
  
  typedef GeometricSearchDet::DetWithState   DetWithState;
  const DetLayer* tkLayer = detLayer(layer,PP.Z());

  TrajectoryStateOnSurface trajState = makeTrajectoryState( tkLayer, PP, &mf);
  float thickness = theMaterialEffects ? theMaterialEffects->thickness() : 0.;
  float eloss = theMaterialEffects ? theMaterialEffects->energyLoss() : 0.;

  // Find, in the corresponding layers, the detectors compatible 
  // with the current track 
  std::vector<DetWithState> compat 
    = tkLayer->compatibleDets( trajState, alongProp, est);

  // And create the corresponding PSimHits
  std::map<double,PSimHit> theTrackHits;
  for (std::vector<DetWithState>::const_iterator i=compat.begin(); i!=compat.end(); i++) {
    // Correct Eloss for last 3 rings of TEC (thick sensors, 0.05 cm)
    // Disgusting fudge factor ! 
    makePSimHits( i->first, i->second, theHitMap, trackID, eloss, thickness, partID,tTopo);
  }

}

TrajectoryStateOnSurface 
TrajectoryManager::makeTrajectoryState( const DetLayer* layer, 
					const ParticlePropagator& pp,
					const MagneticField* field) const
{
  GlobalPoint  pos( pp.X(), pp.Y(), pp.Z());
  GlobalVector mom( pp.Px(), pp.Py(), pp.Pz());
  auto plane = layer->surface().tangentPlane(pos);
  return TrajectoryStateOnSurface
    (GlobalTrajectoryParameters( pos, mom, TrackCharge( pp.charge()), field), *plane);
}

void 
TrajectoryManager::makePSimHits( const GeomDet* det, 
				 const TrajectoryStateOnSurface& ts,
				 std::map<double,PSimHit>& theHitMap,
				 int tkID, float el, float thick, int pID,
				 const TrackerTopology *tTopo) 
{

  std::vector< const GeomDet*> comp = det->components();
  if (!comp.empty()) {
    for (std::vector< const GeomDet*>::const_iterator i = comp.begin();
	 i != comp.end(); i++) {
      auto du = (*i);
      if (du->isLeaf())  // not even needed (or it should iterate if really not leaf)
	theHitMap.insert(theHitMap.end(),makeSinglePSimHit( *du, ts, tkID, el, thick, pID,tTopo));
    }
  }
  else {
    auto du = (det);
    theHitMap.insert(theHitMap.end(),makeSinglePSimHit( *du, ts, tkID, el, thick, pID,tTopo));
  }

}

std::pair<double,PSimHit> 
TrajectoryManager::makeSinglePSimHit( const GeomDetUnit& det,
				      const TrajectoryStateOnSurface& ts, 
				      int tkID, float el, float thick, int pID,
				      const TrackerTopology *tTopo) const
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
      // edm::LogWarning("FastTracking") << "TrajectoryManager ERROR: crossing with det failed, skipping PSimHit";
      return  std::pair<double,PSimHit>(0.,PSimHit());
    }
    lpos = det.toLocal( GlobalPoint( crossing.position(path.second)));
    lmom = det.toLocal( GlobalVector( crossing.direction(path.second)));
    lmom = lmom.unit() * ts.localMomentum().mag();
  }

  // The module (half) thickness 
  const BoundPlane& theDetPlane = det.surface();
  float halfThick = 0.5*theDetPlane.bounds().thickness();
  // The Energy loss rescaled to the module thickness
  float eloss = el;
  if ( thick > 0. ) {
    // Total thickness is in radiation lengths, 1 radlen = 9.36 cm
    // Sensitive module thickness is about 30 microns larger than 
    // the module thickness itself
    eloss *= (2.* halfThick - 0.003) / (9.36 * thick);
  }
  // The entry and exit points, and the time of flight
  float pZ = lmom.z();
  LocalPoint entry = lpos + (-halfThick/pZ) * lmom;
  LocalPoint exit = lpos + halfThick/pZ * lmom;
  float tof = ts.globalPosition().mag() / 30. ; // in nanoseconds, FIXME: very approximate

  // If a hadron suffered a nuclear interaction, just assign the hits of the closest 
  // daughter to the mother's track. The same applies to a charged particle decay into
  // another charged particle.
  int localTkID = tkID;
  if ( !mySimEvent->track(tkID).noMother() && mySimEvent->track(tkID).mother().closestDaughterId() == tkID )
    localTkID = mySimEvent->track(tkID).mother().id();

  // FIXME: fix the track ID and the particle ID
  PSimHit hit( entry, exit, lmom.mag(), tof, eloss, pID,
		  det.geographicalId().rawId(), localTkID,
		  lmom.theta(),
		  lmom.phi());

  // Check that the PSimHit is physically on the module!
  unsigned subdet = DetId(hit.detUnitId()).subdetId(); 
  double boundX = theDetPlane.bounds().width()/2.;
  double boundY = theDetPlane.bounds().length()/2.;

  // Special treatment for TID and TEC trapeziodal modules
  if ( subdet == 4 || subdet == 6 ) 
    boundX *=  1. - hit.localPosition().y()/theDetPlane.position().perp();

#ifdef FAMOS_DEBUG
  unsigned detid  = DetId(hit.detUnitId()).rawId();
  unsigned stereo = 0;
  unsigned theLayer = 0;
  unsigned theRing = 0;
  switch (subdet) { 
  case 1: 
    {
      
      theLayer = tTopo->pxbLayer(detid);
      std::cout << "\tPixel Barrel Layer " << theLayer << std::endl;
      stereo = 1;
      break;
    }
  case 2: 
    {
      
      theLayer = tTopo->pxfDisk(detid);
      std::cout << "\tPixel Forward Disk " << theLayer << std::endl;
      stereo = 1;
      break;
    }
  case 3:
    {
      
      theLayer  = tTopo->tibLayer(detid);
      std::cout << "\tTIB Layer " << theLayer << std::endl;
      stereo = module.stereo();
      break;
    }
  case 4:
    {
      
      theLayer = tTopo->tidWheel(detid);
      theRing  = tTopo->tidRing(detid);
      unsigned int theSide = module.side();
      if ( theSide == 1 ) 
       	std::cout << "\tTID Petal Back " << std::endl; 
      else
	std::cout << "\tTID Petal Front" << std::endl; 
      std::cout << "\tTID Layer " << theLayer << std::endl;
      std::cout << "\tTID Ring " << theRing << std::endl;
      stereo = module.stereo();
      break;
    }
  case 5:
    {
      
      theLayer  = tTopo->tobLayer(detid);
      stereo = tTopo->tobStereo(detid);
      std::cout << "\tTOB Layer " << theLayer << std::endl;
      break;
    }
  case 6:
    {
      
      theLayer = tTopo->tecWheel(detid);
      theRing  = tTopo->tecRing(detid);
      unsigned int theSide = module.petal()[0];
      if ( theSide == 1 ) 
	std::cout << "\tTEC Petal Back " << std::endl; 
      else
	std::cout << "\tTEC Petal Front" << std::endl; 
      std::cout << "\tTEC Layer " << theLayer << std::endl;
      std::cout << "\tTEC Ring " << theRing << std::endl;
      stereo = module.stereo();
      break;
    }
  default:
    {
      stereo = 0;
      break;
    }
  }
  
  std::cout << "Thickness = " << 2.*halfThick-0.003 << "; " << thick * 9.36 << std::endl
	    << "Length    = " << det.surface().bounds().length() << std::endl
	    << "Width     = " << det.surface().bounds().width() << std::endl;
    
  std::cout << "Hit position = " 
	    << hit.localPosition().x() << " " 
	    << hit.localPosition().y() << " " 
	    << hit.localPosition().z() << std::endl;
#endif

  // Check if the hit is on the physical volume of the module
  // (It happens that it is not, in the case of double sided modules,
  //  because the envelope of the gluedDet is larger than each of 
  //  the mono and the stereo modules)

  double dist = 0.;
  GlobalPoint IP (mySimEvent->track(localTkID).vertex().position().x(),
		  mySimEvent->track(localTkID).vertex().position().y(),
		  mySimEvent->track(localTkID).vertex().position().z());

  dist = ( fabs(hit.localPosition().x()) > boundX  || 
	   fabs(hit.localPosition().y()) > boundY ) ?  
    // Will be used later as a flag to reject the PSimHit!
    -( det.surface().toGlobal(hit.localPosition()) - IP ).mag2() 
    : 
    // These hits are kept!
     ( det.surface().toGlobal(hit.localPosition()) - IP ).mag2();

  // Fill Histos (~poor man event display)
  /*  
     GlobalPoint gpos( det.toGlobal(hit.localPosition()));
//      std::cout << "gpos.x() = " << gpos.x() << std::endl;
//      std::cout << "gpos.y() = " << gpos.y() << std::endl;

     myHistos->fill("h300",gpos.x(),gpos.y());
     if ( sin(gpos.phi()) > 0. ) 
     myHistos->fill("h301",gpos.z(),gpos.perp());
     else
     myHistos->fill("h301",gpos.z(),-gpos.perp());
  */
  return std::pair<double,PSimHit>(dist,hit);

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

  const std::vector< const BarrelDetLayer*>&   barrelLayers = 
    theGeomSearchTracker->barrelLayers();
  LogDebug("FastTracking") << "Barrel DetLayer dump: ";
  for (auto bl=barrelLayers.begin();
       bl != barrelLayers.end(); ++bl) {
    LogDebug("FastTracking")<< "radius " << (**bl).specificSurface().radius(); 
  }

  const std::vector< const ForwardDetLayer*>&  posForwardLayers = 
    theGeomSearchTracker->posForwardLayers();
  LogDebug("FastTracking") << "Positive Forward DetLayer dump: ";
  for (auto fl=posForwardLayers.begin();
       fl != posForwardLayers.end(); ++fl) {
    LogDebug("FastTracking") << "Z pos "
			    << (**fl).surface().position().z()
			    << " radii " 
			    << (**fl).specificSurface().innerRadius() 
			    << ", " 
			    << (**fl).specificSurface().outerRadius(); 
  }

  const float rTolerance = 1.5;
  const float zTolerance = 3.;

  LogDebug("FastTracking")<< "Dump of TrackerInteractionGeometry cylinders:";
  for( std::list<TrackerLayer>::const_iterator i=_theGeometry->cylinderBegin();
       i!=_theGeometry->cylinderEnd(); ++i) {
    const BoundCylinder* cyl = i->cylinder();
    const BoundDisk* disk = i->disk();

    LogDebug("FastTracking") << "Famos Layer no " << i->layerNumber()
			    << " is sensitive? " << i->sensitive()
			    << " pos " << i->surface().position();
    if (!i->sensitive()) continue;

    if (cyl != 0) {
      LogDebug("FastTracking") << " cylinder radius " << cyl->radius();
      bool found = false;
      for (auto
	     bl=barrelLayers.begin(); bl != barrelLayers.end(); ++bl) {

	if (fabs( cyl->radius() - (**bl).specificSurface().radius()) < rTolerance) {
	  theLayerMap[i->layerNumber()] = *bl;
	  found = true;
	  LogDebug("FastTracking")<< "Corresponding DetLayer found with radius "
				 << (**bl).specificSurface().radius();
	  break;
	}
      }
      if (!found) {
	edm::LogWarning("FastTracking") << " Trajectory manager FAILED to find a corresponding DetLayer!";
      }
    }
    else {
      LogDebug("FastTracking") << " disk radii " << disk->innerRadius() 
		 << ", " << disk->outerRadius();
      bool found = false;
      for (auto fl=posForwardLayers.begin();
	   fl != posForwardLayers.end(); ++fl) {
	
	if (fabs( disk->position().z() - (**fl).surface().position().z()) < zTolerance) {
	  theLayerMap[i->layerNumber()] = *fl;
	  found = true;
	  LogDebug("FastTracking") << "Corresponding DetLayer found with Z pos "
				  << (**fl).surface().position().z()
				  << " and radii " 
				  << (**fl).specificSurface().innerRadius() 
				  << ", " 
				  << (**fl).specificSurface().outerRadius(); 
	  break;
	}
      }
      if (!found) {
	edm::LogWarning("FastTracking") << "FAILED to find a corresponding DetLayer!";
      }
    }
  }

  // Put the negative layers in the same map but with an offset
 const  std::vector< const ForwardDetLayer*>&  negForwardLayers = theGeomSearchTracker->negForwardLayers();
  for (auto nl=negForwardLayers.begin();
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

  std::map<unsigned,std::map<double,PSimHit> >::const_iterator itrack = thePSimHits.begin();
  std::map<unsigned,std::map<double,PSimHit> >::const_iterator itrackEnd = thePSimHits.end();
  for ( ; itrack != itrackEnd; ++itrack ) {
    std::map<double,PSimHit>::const_iterator it = (itrack->second).begin();
    std::map<double,PSimHit>::const_iterator itEnd = (itrack->second).end();
    for( ; it!= itEnd; ++it) { 
      /*
      DetId theDetUnitId((it->second).detUnitId());
      const GeomDet* theDet = theGeomTracker->idToDet(theDetUnitId);
      std::cout << "Track/z/r after : "
		<< (it->second).trackId() << " " 
		<< theDet->surface().toGlobal((it->second).localPosition()).z() << " " 
		<< theDet->surface().toGlobal((it->second).localPosition()).perp() << std::endl;
      */
      // Keep only those hits that are on the physical volume of a module
      // (The other hits have been assigned a negative <double> value. 
      if ( it->first > 0. ) c.push_back(it->second); 
    }
  }

}
