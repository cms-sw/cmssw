#include "RecoMuon/TrackerSeedGenerator/src/TrackerSeedGenerator.h"

/** \class TrackerSeedGenerator
 *  Generate seed from muon trajectory.
 *
 *  $Date: 2006/09/02 01:02:14 $
 *  $Revision: 1.7 $
 *  \author Norbert Neumeister - Purdue University
 *  \porting author Chang Liu - Purdue University
 */

//---------------
// C++ Headers --
//---------------

#include <string>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
//#include "RecoMuon/TrackerSeedGenerator/interface/PrimitiveMuonSeed.h"
//#include "RecoMuon/TrackerSeedGenerator/interface/MuonSeedFromConsecutiveHits.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixel.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h" 
#include "DataFormats/Common/interface/OwnVector.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

using namespace std;
using namespace edm;

//---------------------------------
//       class TrackerSeedGenerator
//---------------------------------

TrackerSeedGenerator::TrackerSeedGenerator(const edm::ParameterSet& par, const MuonServiceProxy *service) :
  theService(service),
  theVertexPos(GlobalPoint(0.0,0.0,0.0)),
  theVertexErr(GlobalError(0.0001,0.0,0.0001,0.0,0.0,28.09)),
  combinatorialSeedGenerator(par)
{
  ParameterSet updatorPSet = par.getParameter<ParameterSet>("UpdatorParameters");
  theUpdator = new MuonUpdatorAtVertex(updatorPSet,theService);

  theErrorRescale = par.getParameter<double>("ErrorRescaleFactor");
  theOption = par.getParameter<int>("SeedOption");
  hitProducer = par.getParameter<std::string>("HitProducer");
  theMaxSeeds = par.getParameter<int>("MaxSeeds");

  //ParameterSet pixelPSet = par.getParameter<ParameterSet>("PixelParameters");
  //combinatorialSeedGenerator = new CombinatorialSeedGeneratorFromPixel(par);
  //theDirection = static_cast<ReconstructionDirection>(par.getParameter<int>("Direction")); 
  /*
  edm::LogInfo ("TrackSeedGeneratorFromMuon")<<"TrackerSeedGeneratorFromMuon";
  
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  setup.get<IdealMagneticFieldRecord>().get(theField);
  setup.get<TrackerRecoGeometryRecord>().get(theGeoTracker);
  thePropagator = new AnalyticalPropagator(&*theField, oppositeToMomentum);
  theStepPropagator = new SteppingHelixPropagator(&*theField,oppositeToMomentum);


  
  theUseVertex = par.getParameter<bool>("UseVertex");
  
  theMaxLayers = par.getParameter<int>("MaxLayers");

  edm::ParameterSet meastkPar = par.getParameter<edm::ParameterSet>("MeasurementTrackerParameters");
  theMeasurementTracker = new MeasurementTracker(setup,meastkPar);
  theLayerMeasurements = new LayerMeasurements(theMeasurementTracker);
  edm::ParameterSet seedPar = par.getParameter<edm::ParameterSet>("SeedGeneratorParameters");
  theSeedGenerator = new CombinatorialSeedGeneratorFromPixel(seedPar);
  */
}


//----------------
// Destructor   --
//----------------

TrackerSeedGenerator::~TrackerSeedGenerator() {
  if(theUpdator) delete theUpdator;
  //if(combinatorialSeedGenerator) delete combinatorialSeedGenerator;
  /*
    delete theSeedGenerator;
    delete theStepPropagator;
    delete thePropagator;
    delete theLayerMeasurements;
    delete theMeasurementTracker;
  */
}

void TrackerSeedGenerator::setEvent(const edm::Event &event)
{
  event.getByLabel(hitProducer, pixelHits);
}

//
BTSeedCollection TrackerSeedGenerator::trackerSeeds(const Trajectory& muon) {
   theSeeds.clear();
   findSeeds(muon);
   return BTSeedCollection(theSeeds);

}
//
void TrackerSeedGenerator::findSeeds(const Trajectory& muon) {
  
  // track at innermost muon station
  TrajectoryStateOnSurface traj = muon.firstMeasurement().updatedState();
  if ( muon.direction() == oppositeToMomentum ) 
    traj = muon.lastMeasurement().updatedState();
  
  //FIXME: the result of STA should contain Trajectory or something more
  if ( !traj.isValid() ) return;
  
  // propagate to the outer tracker surface (r = 123.3cm, halfLength = 293.5cm)
  //MuonUpdatorAtVertex updator;
  MuonVertexMeasurement vm = theUpdator->update(traj);
  TrajectoryStateOnSurface traj_trak = vm.stateAtTracker();
  
  if ( !traj_trak.isValid() ) return;
  
  // rescale errors
  traj_trak.rescaleError(theErrorRescale);
  
  // find list of possible start layers
  //findLayerList(traj_trak);  
  
  // define tracker region of interest
  GlobalVector mom = traj_trak.globalMomentum();

  TrajectoryStateOnSurface traj_vertex = vm.stateAtVertex();
  if ( traj_vertex.isValid() ) mom = traj_vertex.globalMomentum();
  float eta1   = mom.eta();
  float phi1   = mom.phi();
  float theta1 = mom.theta();

  float eta2 = 0.0;
  float phi2 = 0.0;
  float theta2 = 0.0;
  TransientTrackingRecHit::ConstRecHitContainer recHits = muon.recHits();
  const TransientTrackingRecHit& r = **(recHits.begin()+1);
  eta2   = r.globalPosition().eta();
  phi2   = r.globalPosition().phi();
  theta2 = r.globalPosition().theta();
  
  float deta(fabs(eta1-eta2));
  float dphi(fabs(Geom::Phi<float>(phi1)-Geom::Phi<float>(phi2)));

  double deltaEta = 0.05;
  double deltaPhi = 0.07; // 5*ephi;
  double deltaZ   = min(15.9,3*sqrt(theVertexErr.czz()));
  double minPt    = max(1.5,mom.perp()*0.6);
  
  Geom::Phi<float> phi(phi1);
  Geom::Theta<float> theta(theta1);
  if ( deta > 0.06 ) {
    deltaEta += (deta/2.);
  } 
  if ( dphi > 0.07 ) {
    deltaPhi += 0.15;
    if ( fabs(eta2) < 1.0 && mom.perp() < 6. ) deltaPhi = dphi;
  }
  if ( fabs(eta1) < 1.25 && fabs(eta1) > 0.8 ) deltaEta = max(0.07,deltaEta);
  if ( fabs(eta1) < 1.3  && fabs(eta1) > 1.0 ) deltaPhi = max(0.3,deltaPhi);

  GlobalVector direction(theta,phi,mom.perp());

  RectangularEtaPhiTrackingRegion rectRegion(direction, theVertexPos,
                                             minPt, 0.2, deltaZ, deltaEta, deltaPhi);

    
  // seed options:
  // 0 = primitive seeds  
  // 1 = consecutive seeds 
  // 2 = pixel seeds  
  // 3 = combined (0+1)
  // 4 = combined (2+1)

  switch ( theOption ) {
    case 0 : { primitiveSeeds(muon,traj_trak); break; }
    case 1 : { consecutiveHitsSeeds(muon,traj_trak,theService->eventSetup(),rectRegion); break; }
    case 2 : { pixelSeeds(muon,traj_trak,rectRegion,deltaEta,deltaPhi); break; }
    case 3 : { primitiveSeeds(muon,traj_trak);
               consecutiveHitsSeeds(muon,traj_trak,theService->eventSetup(),rectRegion);
               break; }
    case 4 : { pixelSeeds(muon,traj_trak,rectRegion,deltaEta,deltaPhi);
               consecutiveHitsSeeds(muon,traj_trak,theService->eventSetup(),rectRegion);
               break; }
      /*
	default : { if ( theDirection == outsideIn ) {
	primitiveSeeds(muon,traj_trak);
	if ( theSeeds.size() == 0 ) consecutiveHitsSeeds(muon,traj_trak,theService->eventSetup(),rectRegion);
	}
	else {
	pixelSeeds(muon,traj_trak,rectRegion,deltaEta,deltaPhi);
	if ( theSeeds.size() == 0 ) consecutiveHitsSeeds(muon,traj_trak,theService->eventSetup(),rectRegion);
	}     
	break; }
      */
  }

  

}

//
// get ordered list of tracker layers which may be used as seeds
//
void TrackerSeedGenerator::findLayerList(const TrajectoryStateOnSurface& traj) {
  /*            
  theLayerList.clear();
  // we start from the outer surface of the tracker so it's oppositeToMomentum

  // barrel
  const std::vector<BarrelDetLayer*>& barrel = theGeoTracker->barrelLayers();

  int layercounter = 0;
  for (std::vector<BarrelDetLayer*>::const_iterator ilayer = barrel.begin(); ilayer != barrel.end(); ilayer++ ) {
    layercounter++;
 
    const BoundCylinder& sur = (*ilayer)->specificSurface();    
    TrajectoryStateOnSurface start = thePropagator->propagate(traj,sur);
    if ( !start.isValid() ) continue;
    GlobalPoint gpos = start.globalPosition();
    LocalPoint pos = sur.toLocal(gpos);
    bool in = sur.bounds().inside(pos);
    if ( in ) theLayerList.push_back(MuonSeedDetLayer((*ilayer),gpos,layercounter,start));
  }
  
  // endcaps
  float z = traj.globalPosition().z();
  if ( fabs(z) > 100 ) {
 
    const std::vector<ForwardDetLayer*>& endcap = theGeoTracker->forwardLayers();
 
    layercounter = 0;
    for (std::vector<ForwardDetLayer*>::const_iterator ilayer = endcap.begin(); ilayer != endcap.end(); ilayer++ ) {
      float zl = (*ilayer)->position().z();
      if ( zl*z < 0 ) continue;
      layercounter++;
      if ( fabs(zl) > fabs(z) ) continue;
      const BoundDisk& sur = (*ilayer)->specificSurface();
      TrajectoryStateOnSurface start = thePropagator->propagate(traj,sur);
      if ( !start.isValid() ) continue;
      GlobalPoint gpos = start.globalPosition();
      LocalPoint pos = sur.toLocal(gpos);
      bool in = sur.bounds().inside(pos);
      if ( in ) theLayerList.push_back(MuonSeedDetLayer((*ilayer),gpos,layercounter,start));
    }
  }
  
  // sort layer list
  if ( theDirection == outsideIn ) {              // start from outside
    stable_sort( theLayerList.begin(), theLayerList.end(), MuonSeedDetLayer::LowerR() );
  }
  else if ( theDirection == insideOut ) {         // start from inside
    stable_sort( theLayerList.begin(), theLayerList.end(), MuonSeedDetLayer::HigherR() );
  }
  else edm::LogInfo("TrackerSeedGenerator")<< "Direction Error";

  */  
}

// primitive seeds
//
void TrackerSeedGenerator::primitiveSeeds(const Trajectory& muon,
					  const TrajectoryStateOnSurface& traj) {
  /*
  int nseeds = theSeeds.size();
  int nlayers = 0;
  bool validLayer = false;
  
  // define propagation direction
  PropagationDirection dir = alongMomentum;
  if ( theDirection == outsideIn ) {              // start from outside
    dir = oppositeToMomentum;
  }
  else if ( theDirection == insideOut ) {         // start from inside
    dir = alongMomentum;
  }
  
  vector<MuonSeedDetLayer>::const_iterator ilayer;
  for ( ilayer = theLayerList.begin(); ilayer != theLayerList.end(); ilayer++ ) { // start from theDirection
    const DetLayer* layer = (*ilayer).layer(); 
    const TrajectoryStateOnSurface start = (*ilayer).state();
  
    if ( validLayer ) nlayers++;
    if ( nlayers >= theMaxLayers ) break;
    validLayer = false;

    if ( start.isValid() ) {
      // find measurements on layer
      double maxChi2 = 150.0;
      Chi2MeasurementEstimator aEstimator(maxChi2);
      
      const std::vector<TrajectoryMeasurement> meas = 
	theLayerMeasurements->measurements((*layer),start,*thePropagator,aEstimator); 
      //?FIXME: no fast version for layer
      
      vector<TrajectoryMeasurement>::const_iterator it;
      for ( it = meas.begin(); it != meas.end(); it++ ) {
        if ( (*it).recHit()->isValid() ) {
          TrajectoryStateTransform tsTransform;
          PTrajectoryStateOnDet* ptsos = tsTransform.persistentState(start,(*it).recHit()->geographicalId().rawId());
	  
	  edm::OwnVector< TrackingRecHit > layerRecHits;
          layerRecHits.push_back( ((*it).recHit()->hit()->clone()) ); // start from theDirection
	  
	  //PrimitiveMuonSeed* theSeed = new PrimitiveMuonSeed(*ptsos,dir,*it);
	  PrimitiveMuonSeed* theSeed = new PrimitiveMuonSeed(*ptsos,dir,layerRecHits,*it);
          validLayer = true;
	  
          if ( nlayers < theMaxLayers && nseeds < theMaxSeeds ) {
            theSeeds.push_back(theSeed);
            nseeds++;
          }
          else {
            delete theSeed;
            break;
          }
          
        }  
      }
    }
  }
  */
}


//
// seeds from consecutive hits
//
void TrackerSeedGenerator::consecutiveHitsSeeds(const Trajectory& muon,
						const TrajectoryStateOnSurface& traj,
						const edm::EventSetup& iSetup,
                                                const TrackingRegion& regionOfInterest) {
  /*
  if ( theLayerList.size() < 2 ) return;

  int nlayers = 0;
  int nseeds = theSeeds.size();

  vector<MuonSeedDetLayer>::const_iterator layer1,layer2;
  for ( layer1 = theLayerList.begin(); layer1 != theLayerList.end()-1; layer1++ ) { // first layer in theDirection
    if ( nlayers >= theMaxLayers || nseeds >= theMaxSeeds ) break;
    for ( layer2 = layer1+1; layer2 != theLayerList.end(); layer2++ ) { // second layer in theDirection
      if ( theDirection == outsideIn ) {  // start from outside
        createSeed(*layer1,*layer2,iSetup,regionOfInterest);
      }
      if ( theDirection == insideOut ) {  // start from innside
        createSeed(*layer2,*layer1,iSetup,regionOfInterest);
      }
    }
    int newseeds = theSeeds.size();
    if ( newseeds > nseeds ) {
      nlayers++;
      nseeds = newseeds;
    }  
  }
  */
}


//
// create seeds from consecutive hits
//
void TrackerSeedGenerator::createSeed(const MuonSeedDetLayer& outer,
                                      const MuonSeedDetLayer& inner,
                                      const edm::EventSetup& iSetup,
                                      const TrackingRegion& regionOfInterest) {
  /*
  int nseeds = theSeeds.size();
  
  PropagationDirection dir = alongMomentum;
  if ( theDirection == outsideIn ) {              // start from outside
    dir = oppositeToMomentum;
  }
  if ( theDirection == insideOut ) {              // start from innside
    dir = alongMomentum;
  }

  const DetLayer* outerlayer = outer.layer();
  const DetLayer* innerlayer = inner.layer();

  TrajectoryStateOnSurface start1 = outer.state();
  TrajectoryStateOnSurface start2 = inner.state();
  
  if ( start1.isValid() && start2.isValid() ) {
    // find measurements on layer
    double maxChi2 = 150.0;
    Chi2MeasurementEstimator aEstimator(maxChi2);

    //?FIXME: no fast version for layer    
    const std::vector<TrajectoryMeasurement> measA = 
      theLayerMeasurements->measurements((*outerlayer),start1,*thePropagator,aEstimator); 
    const std::vector<TrajectoryMeasurement> measB = 
      theLayerMeasurements->measurements((*innerlayer),start2,*thePropagator,aEstimator); 
    
    //?FIXME method not implemented in TrackingRegion    
    //const std::vector<TransientTrackingRecHit> meas1;// = regionOfInterest.hits(outerlayer);
    //const std::vector<TransientTrackingRecHit> meas2;// = regionOfInterest.hits(innerlayer);
    
    TransientTrackingRecHit::RecHitContainer layerRecHitsA;
    TransientTrackingRecHit::RecHitContainer layerRecHitsB;
    
    vector<TrajectoryMeasurement>::const_iterator it;
    for ( it = measA.begin(); it != measA.end(); it++ ) {
      if ( (*it).recHit()->isValid() ) {
	const HitRZCompatibility *checkRZ = regionOfInterest.checkRZ(&(*outerlayer),&*(*it).recHit(),iSetup);
	if(!checkRZ) continue;
	if((*checkRZ)( (*it).recHit()->globalPosition().perp(), (*it).recHit()->globalPosition().z())) {
	  layerRecHitsA.push_back( ((*it).recHit()) ); // start from theDirection
	}
      }
    }
    for ( it = measB.begin(); it != measB.end(); it++ ) {
      if ( (*it).recHit()->isValid() ) {
	const HitRZCompatibility *checkRZ = regionOfInterest.checkRZ(&(*innerlayer),&*(*it).recHit(),iSetup);
	if(!checkRZ) continue;
	if((*checkRZ)( (*it).recHit()->globalPosition().perp(), (*it).recHit()->globalPosition().z())) {	  
	  layerRecHitsB.push_back( (*it).recHit() ); // start from theDirection
	}
      }
    }
 
    TransientTrackingRecHit::RecHitContainer::const_iterator it1,it2;
    for ( it1 = layerRecHitsA.begin(); it1 != layerRecHitsA.end(); it1++ ) {
      if ( !(**it1).isValid() ) continue;
      for ( it2 = layerRecHitsB.begin(); it2 != layerRecHitsB.end(); it2++ ) {
	if ( !(**it2).isValid() ) continue;
	MuonSeedFromConsecutiveHits* seed = new MuonSeedFromConsecutiveHits((**it1),(**it2),dir,theVertexPos, theVertexErr, iSetup);
	if ( seed->recHits().first == seed->recHits().second ) {
	  delete seed;
	}
	else if ( fabs(seed->freeTrajectoryState().momentum().eta()-regionOfInterest.direction().eta()) > 0.1 ) {
	  delete seed;
	}
	else {
	  if ( nseeds < theMaxSeeds ) {
	    theSeeds.push_back(seed);
	    nseeds++;
	  }
	  else {
	    delete seed;
	    break;
	  } 
	}
      }
    }   
  }
  */
}


//
// seeds from pixels
//
void TrackerSeedGenerator::pixelSeeds(const Trajectory& muon, 
                                      const TrajectoryStateOnSurface& traj,
                                      const RectangularEtaPhiTrackingRegion& regionOfInterest,
                                      float deltaEta, float deltaPhi) {
  
  //std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  vector<TrajectorySeed> ss;
  
  RectangularEtaPhiTrackingRegion region = RectangularEtaPhiTrackingRegion(regionOfInterest);
  
  combinatorialSeedGenerator.init(*pixelHits,theService->eventSetup());
  combinatorialSeedGenerator.run(region,ss,theService->eventSetup());
  
  int nseeds = theSeeds.size();
  vector<TrajectorySeed>::const_iterator is;
  for ( is = ss.begin(); is != ss.end(); is++ ) {
    if ( nseeds < theMaxSeeds ) {
      theSeeds.push_back(const_cast<TrajectorySeed*>(&((*is)))); 
      nseeds++;
    }
  }

  /*
    int nseeds = theSeeds.size();
    TrajectoryStateTransform tsTransform;
    vector<TrajectorySeed> ss;
    theSeedGenerator->seeds(ss, theService->eventSetup(), regionOfInterest);
    vector<TrajectorySeed>::const_iterator is;
    for ( is = ss.begin(); is != ss.end(); is++ ) {
    PTrajectoryStateOnDet ptsos = (*is).startingState();
    const GeomDet* gdet = theTrackingGeometry->idToDet(DetId(ptsos.detId()));
    TrajectoryStateOnSurface tsos = tsTransform.transientState(ptsos, &(gdet->surface()), &*theField);
    float eta = tsos.globalMomentum().eta();
    float phi = tsos.globalMomentum().phi();
    float deta(fabs(eta-regionOfInterest.direction().eta()));
    float dphi(fabs(Geom::Phi<float>(phi)-Geom::Phi<float>(regionOfInterest.direction().phi())));
    if ( deta > deltaEta || dphi > deltaPhi ) continue;     
    if ( nseeds < theMaxSeeds ) {
    theSeeds.push_back(const_cast<TrajectorySeed*>(&((*is)))); 
    nseeds++;
    }
    else {
    break;
    } 
    }  
  */
}
