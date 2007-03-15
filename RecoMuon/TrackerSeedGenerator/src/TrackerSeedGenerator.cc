#include "RecoMuon/TrackerSeedGenerator/src/TrackerSeedGenerator.h"

/** \class TrackerSeedGenerator
 *  Generate seed from muon trajectory.
 *
 *  $Date: 2007/03/14 11:40:20 $
 *  $Revision: 1.17 $
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
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
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
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoMuon/GlobalMuonProducer/src/GlobalMuonMonitorInterface.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"

//#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

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
  theUpdator = new MuonUpdatorAtVertex(par.getParameter<string>("UpdatorPropagator"),theService);

  theErrorRescale = par.getParameter<double>("ErrorRescaleFactor");
  theOption = par.getParameter<int>("SeedOption");
  hitProducer = par.getParameter<string>("HitProducer");
  theMaxSeeds = par.getParameter<int>("MaxSeeds");

  double roadChi2 = par.getParameter<double>("maxRoadChi2");
  theRoadEstimator = new Chi2MeasurementEstimator(roadChi2,sqrt(roadChi2));
  theOutPropagator = par.getParameter<string>("StateOnTrackerBoundOutPropagator");
  theRSAlongPropagator = par.getParameter<string>("RSAlongPropagator");
  theRSAnyPropagator = par.getParameter<string>("RSAnyPropagator");

  theMIMFlag = par.getUntrackedParameter<bool>("performMuonIntegrityMonitor",false);
  if(theMIMFlag) {
    dataMonitor = edm::Service<GlobalMuonMonitorInterface>().operator->(); 
  }
}


//----------------
// Destructor   --
//----------------

TrackerSeedGenerator::~TrackerSeedGenerator() {
  if(theUpdator) delete theUpdator;
  if(theRoadEstimator) delete theRoadEstimator;
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
  theService->eventSetup().get<TrackerRecoGeometryRecord>().get(theSearchTracker);
}

/*!
 * \param muon the standalone muon track
 * \param rectRegion a RectangularEtaPhiTracking region defined around the muon
 */
TrackerSeedGenerator::BTSeedCollection 
TrackerSeedGenerator::trackerSeeds(const TrackCand& muon, const RectangularEtaPhiTrackingRegion& rectRegion) {
   theSeeds.clear();
   findSeeds(muon,rectRegion);
   return BTSeedCollection(theSeeds);

}
//
void TrackerSeedGenerator::findSeeds(const TrackCand& muon, const RectangularEtaPhiTrackingRegion& rectRegion) {

  //if ( !muon.first->isValid() ) return;

  // track at innermost muon station
  TrajectoryStateTransform tsTransform;
  FreeTrajectoryState muFTS = tsTransform.initialFreeState(*(muon.second),&*theService->magneticField());

  //FIXME: the result of STA should contain Trajectory or something more
  //if ( !muFTS.isValid() ) return;

  // Get the Tracker bounds, since the propagation goes from the vertex to the tracker
  // it is always along momentum
  StateOnTrackerBound tracker( &*theService->propagator(theOutPropagator) );
  
  // Get the state at the tracker bound
  TrajectoryStateOnSurface traj_trak = tracker(muFTS);
  
  if ( !traj_trak.isValid() ) return;

  // rescale errors
  traj_trak.rescaleError(theErrorRescale);
  
  // find list of possible start layers
  //findLayerList(traj_trak);  

  double deltaEta = rectRegion.etaRange().max();
  double deltaPhi = rectRegion.phiMargin().right();
    
  // seed options:
  // 0 = primitive seeds  
  // 1 = consecutive seeds 
  // 2 = pixel seeds  
  // 3 = combined (0+1)
  // 4 = combined (2+1)
  // 5 = roadSearch seed

  switch ( theOption ) {
  case 0 : { primitiveSeeds(*muon.first,traj_trak); break; }
  case 1 : { consecutiveHitsSeeds(*muon.first,traj_trak,theService->eventSetup(),rectRegion); break; }
  case 2 : { pixelSeeds(*muon.first,traj_trak,rectRegion,deltaEta,deltaPhi); break; }
  case 3 : { primitiveSeeds(*muon.first,traj_trak);
  consecutiveHitsSeeds(*muon.first,traj_trak,theService->eventSetup(),rectRegion);
  break; }
  case 4 : { pixelSeeds(*muon.first,traj_trak,rectRegion,deltaEta,deltaPhi);
  consecutiveHitsSeeds(*muon.first,traj_trak,theService->eventSetup(),rectRegion);
  break; }
  case 5 : { rsSeeds(*muon.second); break;}
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
  const vector<BarrelDetLayer*>& barrel = theGeoTracker->barrelLayers();

  int layercounter = 0;
  for (vector<BarrelDetLayer*>::const_iterator ilayer = barrel.begin(); ilayer != barrel.end(); ilayer++ ) {
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
 
    const vector<ForwardDetLayer*>& endcap = theGeoTracker->forwardLayers();
 
    layercounter = 0;
    for (vector<ForwardDetLayer*>::const_iterator ilayer = endcap.begin(); ilayer != endcap.end(); ilayer++ ) {
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
      
      const vector<TrajectoryMeasurement> meas = 
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
    const vector<TrajectoryMeasurement> measA = 
      theLayerMeasurements->measurements((*outerlayer),start1,*thePropagator,aEstimator); 
    const vector<TrajectoryMeasurement> measB = 
      theLayerMeasurements->measurements((*innerlayer),start2,*thePropagator,aEstimator); 
    
    //?FIXME method not implemented in TrackingRegion    
    //const vector<TransientTrackingRecHit> meas1;// = regionOfInterest.hits(outerlayer);
    //const vector<TransientTrackingRecHit> meas2;// = regionOfInterest.hits(innerlayer);
    
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
  
  //auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  vector<TrajectorySeed> ss;
  
  RectangularEtaPhiTrackingRegion region = RectangularEtaPhiTrackingRegion(regionOfInterest);

  if(theMIMFlag) {
    dataMonitor->book2D("region","Center of Tracking Region",50,-2.5,2.5,50,-3.14,3.14);
    dataMonitor->fill2("region",region.direction().eta(),region.direction().phi());
    dataMonitor->book2D("pixel","Location of First Pixel",50,-2.5,2.5,50,-3.14,3.14);

    dataMonitor->book2D("regionSize","Size of Region",30,0,3,30,0,3);

    float dEta = region.etaRange().max() - region.etaRange().mean();
    float dPhi = region.phiMargin().right();
    dataMonitor->fill2("regionSize",dEta,dPhi);

    dataMonitor->book1D("pixelHits","Number of PixelHits",21,-0.5,20.5);
    dataMonitor->fill1("pixelHits",(*pixelHits).size());
    
    if( (*pixelHits).size() > 0 && (*pixelHits).begin()->isValid()) {
      DetId ida((*pixelHits).begin()->geographicalId());
      float eta1 = (theService->trackingGeometry())->idToDet(ida)->surface().toGlobal((*pixelHits).begin()->localPosition()).eta();
      float phi1 = (theService->trackingGeometry())->idToDet(ida)->surface().toGlobal((*pixelHits).begin()->localPosition()).phi();
            dataMonitor->fill2("pixel",eta1,phi1);
    }    
  }

  combinatorialSeedGenerator.init(*pixelHits,theService->eventSetup());
  combinatorialSeedGenerator.run(region,ss,theService->eventSetup());
  
  typedef edm::OwnVector< TrackingRecHit > 	recHitContainer;
  typedef recHitContainer::const_iterator 	const_iterator;
  typedef std::pair< const_iterator, const_iterator >  range;
  
  vector<int> mask(ss.size(),1);
  
  int seed1 = 0;
  int nseeds = theSeeds.size();

  vector<TrajectorySeed>::const_iterator is;
  for ( is = ss.begin(); is != ss.end(); ++is ) { 
    range rhits = (*is).recHits();
    if(!((*rhits.first).isValid()&&(*(rhits.second-1)).isValid())) continue;
    
    const GlobalPoint& pos1 =  theService->trackingGeometry()->idToDet((*rhits.first).geographicalId())->surface().toGlobal((*rhits.first).localPosition());
    
    const GlobalPoint& pos2 =  theService->trackingGeometry()->idToDet((*(rhits.second-1)).geographicalId())->surface().toGlobal((*(rhits.second-1)).localPosition());
    
    const GlobalVector& vect1(pos2 - pos1);      

    vector<TrajectorySeed>::const_iterator is2;
    for ( is2 = is+1; is2 != ss.end(); ++is2 ) {
      int seed2 = seed1+1;	
      range rhits2 = (*is2).recHits();
      if( !((*rhits2.first).isValid() && (*(rhits2.second -1)).isValid())) continue;
      
      const GlobalPoint& pos1a =  theService->trackingGeometry()->idToDet((*rhits2.first).geographicalId())->surface().toGlobal((*rhits2.first).localPosition());
      
      const GlobalPoint& pos2a =  theService->trackingGeometry()->idToDet((*(rhits2.second-1)).geographicalId())->surface().toGlobal((*(rhits2.second-1)).localPosition());
      
      const GlobalVector& vect2(pos2a - pos1a);      

      //if(pos2.perp() == pos1a.perp() || pos1.perp() == pos2a.perp()) {      
      double dot = vect1.unit().dot(vect2.unit());
      if(dot > 0.99999) mask[seed2]=0;
      //}
      
    }
    seed1++;
  }
  
  int seed = 0;
  for ( is = ss.begin(); is != ss.end(); ++is ) { 
    if(nseeds < theMaxSeeds) {
      if(mask[seed]) theSeeds.push_back(*is); 
      nseeds++;
      seed++;
    }      
  }

}

std::vector<TrajectorySeed> TrackerSeedGenerator::rsSeeds(const reco::Track& muon) {
  const string category = "TrackerSeedGenerator";

  //default result
  std::vector<TrajectorySeed> result;
  
  //propagate it to the IP and combine
  reco::TransientTrack muonTT(muon,&*theService->magneticField(),theService->trackingGeometry());
  FreeTrajectoryState cIPFTS = muonTT.initialFreeState();

  //take state at inner surface and check the first part reached
  vector<BarrelDetLayer*> blc = theSearchTracker->tibLayers();
  TrajectoryStateOnSurface inner = theService->propagator(theRSAlongPropagator)->propagate(cIPFTS,blc[0]->surface());
  if ( !inner.isValid() ) {
    LogDebug(category) <<"inner state is not valid"; 
    return result;
  }
  
  double z = inner.globalPosition().z();


  
  vector<ForwardDetLayer*> ptidc = theSearchTracker->posTidLayers();
  vector<ForwardDetLayer*> ptecc = theSearchTracker->posTecLayers();
  vector<ForwardDetLayer*> ntidc = theSearchTracker->negTidLayers();
  vector<ForwardDetLayer*> ntecc = theSearchTracker->negTecLayers();

  const DetLayer *inLayer = NULL;
  if( fabs(z) < ptidc[0]->surface().position().z()  ) {
    inLayer = blc[0];
  } else if ( fabs(z) < ptecc[0]->surface().position().z() ) {
    inLayer = ( z < 0 ) ? ntidc[0] : ptidc[0] ;
  } else {
    inLayer = ( z < 0 ) ? ntecc[0] : ptecc[0] ;
  }

  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible = inLayer->compatibleDets(inner,*theService->propagator(theRSAnyPropagator),*theRoadEstimator);
  
  //loop the parts until at least a compatible is found
  while (compatible.size()==0) {
    switch ( inLayer->subDetector() ) {
    case PixelSubdetector::PixelBarrel:
    case PixelSubdetector::PixelEndcap:
    case StripSubdetector::TOB:
    case StripSubdetector::TEC:
      return result;
      break;
    case StripSubdetector::TIB:
      inLayer = ( z < 0 ) ? ntidc[0] : ptidc[0] ;
      break;
    case StripSubdetector::TID:
      inLayer = ( z < 0 ) ? ntecc[0] : ptecc[0] ;
      break;
    }
    compatible = inLayer->compatibleDets(inner,*theService->propagator(theRSAnyPropagator),*theRoadEstimator);
  }

  //transform it into a PTrajectoryStateOnDet
  TrajectoryStateTransform tsTransform;
  PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(compatible[0].second,compatible[0].first->geographicalId().rawId());
  
  BasicTrajectorySeed::recHitContainer rhContainer;
  //copy the muon rechit into the seed
  for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
    rhContainer.push_back( (*trit).get()->clone() );
  }
  
  //add this seed to the list and return it
  result.push_back(TrajectorySeed(PTSOD,rhContainer,alongMomentum));

  int nseeds = theSeeds.size();
  vector<TrajectorySeed>::const_iterator is;
  for (is = result.begin(); is != result.end(); ++is) {
    if ( nseeds < theMaxSeeds ) {
      theSeeds.push_back(*is);
      nseeds++;
    }
  }

  return result;

}
