#include "RecoMuon/TrackerSeedGenerator/interface/TSGForRoadSearch.h"

#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>
//#include <RecoTracker/Record/interface/TrackerRecoGeometryRecord.h>
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>
#include <TrackingTools/Records/interface/TrackingComponentsRecord.h>

#include <TrackingTools/TransientTrack/interface/TransientTrack.h>
#include <TrackingTools/DetLayers/interface/BarrelDetLayer.h>
#include <TrackingTools/DetLayers/interface/ForwardDetLayer.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>

#include <RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h>

#include "RecoTracker/TkNavigation/interface/StartingLayerFinder.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include <TrackingTools/KalmanUpdators/interface/KFUpdator.h>

TSGForRoadSearch::TSGForRoadSearch(const edm::ParameterSet & par){

  theOption = par.getParameter<uint>("option");
  theCopyMuonRecHit = par.getParameter<bool>("copyMuonRecHit");

  double Chi2 = par.getParameter<double>("maxChi2");
  if (Chi2>0){ theChi2Estimator = new Chi2MeasurementEstimator(Chi2,sqrt(Chi2));}
  else { theChi2Estimator=0;}
  
  thePropagatorName = par.getParameter<std::string>("propagatorName");
  thePropagatorCompatibleName = par.getParameter<std::string>("propagatorCompatibleName");

  theCategory = "TSGForRoadSearch|TrackerSeedGenerator";
  //  theLayerShift = par.getParameter<uint>("layerShift");

  theManySeeds = par.getParameter<bool>("manySeeds");
  if (theManySeeds){ theUpdator = new KFUpdator();}
  else{  theUpdator=0;}

}
TSGForRoadSearch::~TSGForRoadSearch(){
  delete theChi2Estimator;
  if (theUpdator)  delete theUpdator;
}


void TSGForRoadSearch::init(const MuonServiceProxy* service){
  theProxyService = service;
}

void TSGForRoadSearch::setEvent(const edm::Event &event){
  //get the measurementtracker
  theProxyService->eventSetup().get<CkfComponentsRecord>().get(theMeasurementTracker);
  if (!theMeasurementTracker.isValid())/*abort*/{edm::LogError(theCategory)<<"measurement tracker geometry not found ";}

  //get a kF updator
  //  theProxyService->eventSetup().get<>().get(theUpdator,theUpdatorName);
  //  if (!theUpdator.isValid())/*abort*/{edm::LogError(theCategory)<<" updator is not found";}

}


void  TSGForRoadSearch::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region, std::vector<TrajectorySeed> & result){
  switch (theOption){
  case 0:
    makeSeeds_0(*muonTrackCand.second,result);break;
  case 1:
    makeSeeds_1(*muonTrackCand.second,result);break;
  case 2:
    makeSeeds_2(*muonTrackCand.second,result);break;
  case 3:
    makeSeeds_3(*muonTrackCand.second,result);break;
  case 4:
    makeSeeds_4(*muonTrackCand.second,result);break;
  }  
}



void TSGForRoadSearch::makeSeeds_2(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  /*
  //define a seed on the outer layers, state is on outer layer and direction is oppositeToMomentum
  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(theCategory)<<cIPFTS;
  if (cIPFTS.position().mag()==0)  { edm::LogError(theCategory)<<"initial point of muon is (0,0,0)."; return;} 


  //what are the first layers uncountered
  StartingLayerFinder theFinder(theProxyService->propagator(thePropagatorName),theMeasurementTracker.product());
  std::vector<StartingLayerFinder::LayerWithState> layers = theFinder.startingOuterStripLayerWithStates(cIPFTS);
  LogDebug(theCategory)<<"("<<layers.size()<<") starting layers found";

  std::vector< DetLayer::DetWithState > compatible;
  //loop over them to find the first compatible detector
  for (std::vector<StartingLayerFinder::LayerWithState>::iterator itLWS=layers.begin(); itLWS!=layers.end();++itLWS){
    compatible=itLWS->first->compatibleDets(itLWS->second,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);
    if (!compatible.empty()) break;}
  if (compatible.empty()) {LogDebug(theCategory)<<"no compatible hits."; return;}
  LogDebug(theCategory)<<"("<<compatible.size()<<") compatible dets found";

  pushTrajectorySeed(muon,compatible,oppositeToMomentum,result);

*/
  return;
}

void TSGForRoadSearch::makeSeeds_1(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  /*
  //define a seed on the inner layers, state is on inner layer and direction is alongMomentum

  LogDebug(theCategory)<<"initial state";

  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(theCategory)<<cIPFTS;
  if (cIPFTS.position().mag()==0) { edm::LogError(theCategory)<<"initial point of muon is (0,0,0)."; return;}

  //what are the first layers uncountered
  StartingLayerFinder theFinder(theProxyService->propagator(thePropagatorName),theMeasurementTracker.product());
  std::vector<StartingLayerFinder::LayerWithState> layers = theFinder.startingStripLayerWithStates(cIPFTS);
  LogDebug(theCategory)<<"("<<layers.size()<<") starting layers found";

  std::vector< DetLayer::DetWithState > compatible;
  //loop over them to find the first compatible detector
  for (std::vector<StartingLayerFinder::LayerWithState>::iterator itLWS=layers.begin(); itLWS!=layers.end();++itLWS){
    compatible=itLWS->first->compatibleDets(itLWS->second,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);
    if (!compatible.empty()) break;}
  if (compatible.empty()) {LogDebug(theCategory)<<"no compatible hits."; return;}
  LogDebug(theCategory)<<"("<<compatible.size()<<") compatible dets found";


  pushTrajectorySeed(muon,compatible,alongMomentum,result);

*/  
  return;
}


void TSGForRoadSearch::makeSeeds_0(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(theCategory)<<cIPFTS;   

  //take state at inner surface and check the first part reached
  std::vector<BarrelDetLayer*> blc = theMeasurementTracker->geometricSearchTracker()->tibLayers();
  TrajectoryStateOnSurface inner = theProxyService->propagator(thePropagatorName)->propagate(cIPFTS,blc.front()->surface());
  if ( !inner.isValid() ) {LogDebug(theCategory) <<"inner state is not valid"; return;}

  double z = inner.globalPosition().z();

  std::vector<ForwardDetLayer*> ptidc = theMeasurementTracker->geometricSearchTracker()->posTidLayers();
  std::vector<ForwardDetLayer*> ptecc = theMeasurementTracker->geometricSearchTracker()->posTecLayers();
  std::vector<ForwardDetLayer*> ntidc = theMeasurementTracker->geometricSearchTracker()->negTidLayers();
  std::vector<ForwardDetLayer*> ntecc = theMeasurementTracker->geometricSearchTracker()->negTecLayers();

  const DetLayer *inLayer = NULL;
  if( fabs(z) < ptidc.front()->surface().position().z()  ) {
    inLayer = blc.front();
  } else if ( fabs(z) < ptecc.front()->surface().position().z() ) {
    inLayer = ( z < 0 ) ? ntidc.front() : ptidc.front() ;
  } else {
    inLayer = ( z < 0 ) ? ntecc.front() : ptecc.front() ;
  }

  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible = inLayer->compatibleDets(inner,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);

  //loop the parts until at least a compatible is found
  while (compatible.size()==0) {
    switch ( inLayer->subDetector() ) {
    case PixelSubdetector::PixelBarrel:
    case PixelSubdetector::PixelEndcap:
    case StripSubdetector::TOB:
    case StripSubdetector::TEC:
      return;
      break;
    case StripSubdetector::TIB:
      inLayer = ( z < 0 ) ? ntidc.front() : ptidc.front() ;
      break;
    case StripSubdetector::TID:
      inLayer = ( z < 0 ) ? ntecc.front() : ptecc.front() ;
      break;
    }
    compatible = inLayer->compatibleDets(inner,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);
  }

  pushTrajectorySeed(muon,compatible,alongMomentum,result);

  return;
}


void TSGForRoadSearch::makeSeeds_3(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(theCategory)<<cIPFTS;   
  if (cIPFTS.position().mag()==0) /*error*/ { edm::LogError(theCategory)<<"initial point of muon is (0,0,0)."; return;}

  //take state at inner surface and check the first part reached
  std::vector<BarrelDetLayer*> blc = theMeasurementTracker->geometricSearchTracker()->tobLayers();
  TrajectoryStateOnSurface outer = theProxyService->propagator(thePropagatorName)->propagate(cIPFTS,blc.back()->surface());
  if ( !outer.isValid() ) {LogDebug(theCategory) <<"outer state is not valid"; return;}

  double z = outer.globalPosition().z();

  std::vector<ForwardDetLayer*> ptidc = theMeasurementTracker->geometricSearchTracker()->posTidLayers();
  std::vector<ForwardDetLayer*> ptecc = theMeasurementTracker->geometricSearchTracker()->posTecLayers();
  std::vector<ForwardDetLayer*> ntidc = theMeasurementTracker->geometricSearchTracker()->negTidLayers();
  std::vector<ForwardDetLayer*> ntecc = theMeasurementTracker->geometricSearchTracker()->negTecLayers();

  uint layerShift=0;
  const DetLayer *inLayer = NULL;
  if (fabs(z) < ptecc.front()->surface().position().z()  ){
    inLayer = *(blc.rbegin()+layerShift);
  } else {
    //whoa ! +1 should not be allowed !
    uint tecIt=1;
    for (; tecIt!=ptecc.size();tecIt++){
      if (fabs(z) < ptecc[tecIt]->surface().position().z())
	{inLayer = ( z < 0 ) ? ntecc[tecIt-1] : ptecc[tecIt-1] ; break;}}
    if (!inLayer) {inLayer = ( z < 0 ) ? ntecc.back() : ptecc.back();}
  }

  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible = inLayer->compatibleDets(outer,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);

  //loop the parts until at least a compatible is found
  while (compatible.size()==0) {
    switch ( inLayer->subDetector() ) {
    case PixelSubdetector::PixelBarrel:
    case PixelSubdetector::PixelEndcap:
    case StripSubdetector::TIB:
    case StripSubdetector::TID:
    case StripSubdetector::TOB:
      layerShift++;
      if (layerShift>=blc.size()) return;
      inLayer = *(blc.rbegin()+layerShift);
      break;
    case StripSubdetector::TEC:
      inLayer = *(blc.rbegin()+layerShift);
      break;
    }
    compatible = inLayer->compatibleDets(outer,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);
  }

  pushTrajectorySeed(muon,compatible,oppositeToMomentum,result);

  return;
}


void TSGForRoadSearch::makeSeeds_4(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(theCategory)<<cIPFTS;   

  //take state at inner surface and check the first part reached
  std::vector<BarrelDetLayer*> blc = theMeasurementTracker->geometricSearchTracker()->pixelBarrelLayers();
  if (blc.empty()){edm::LogError(theCategory)<<"want to start from pixel layer, but no barrel exists"; return;}

  TrajectoryStateOnSurface inner = theProxyService->propagator(thePropagatorName)->propagate(cIPFTS,blc.front()->surface());
  if ( !inner.isValid() ) {LogDebug(theCategory) <<"inner state is not valid"; return;}

  double z = inner.globalPosition().z();

  std::vector<ForwardDetLayer*> ppxlc = theMeasurementTracker->geometricSearchTracker()->posPixelForwardLayers();
  std::vector<ForwardDetLayer*> npxlc = theMeasurementTracker->geometricSearchTracker()->negPixelForwardLayers();

  if (ppxlc.empty() || npxlc.empty())
    { edm::LogError(theCategory)<<"want to start from pixel layer, but no forward layer exists"; return;}

  const DetLayer *inLayer = NULL;
  
  const double epsilon = 0.5 ;//cm
  double barrel_half_length = blc.front()->specificSurface().bounds().length()/2. - epsilon;

  if (fabs(z-blc.front()->surface().position().z()) < barrel_half_length)
    {
      inLayer = blc.front();
    }
  else 
    {
      inLayer = ( z < 0 ) ? npxlc.front() : ppxlc.front() ;
    }

  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible = inLayer->compatibleDets(inner,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);

  pushTrajectorySeed(muon,compatible,alongMomentum,result);

  return;
}


#include <TrackingTools/PatternTools/interface/TrajectoryMeasurement.h>
#include <TrackingTools/MeasurementDet/interface/MeasurementDet.h>

void TSGForRoadSearch::pushTrajectorySeed(const reco::Track & muon, std::vector<DetLayer::DetWithState > & compatible, PropagationDirection direction, std::vector<TrajectorySeed>& result)const {

  if (compatible.empty()){
    LogDebug(theCategory)<<"pushTrajectorySeed with no compatible module. 0 seed.";
    return;}

  if (theManySeeds){
    TrajectoryStateTransform tsTransform;    

    //finf out every compatible measurements
    for (std::vector<DetLayer::DetWithState >::iterator DWSit = compatible.begin(); DWSit!=compatible.end();++DWSit){
      bool aBareTS=false;
      const GeomDet * gd = DWSit->first;
      if (!gd){edm::LogError(theCategory)<<"GeomDet is not valid."; continue;}
      const MeasurementDet * md= theMeasurementTracker->idToDet(gd->geographicalId());
      std::vector<TrajectoryMeasurement> tmp = md->fastMeasurements(DWSit->second,DWSit->second,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);
      //make a trajectory seed for each of them

      for (std::vector<TrajectoryMeasurement>::iterator Mit = tmp.begin(); Mit!=tmp.end();++Mit){
	TrajectoryStateOnSurface predState(Mit->predictedState());
	TrajectoryMeasurement::ConstRecHitPointer hit = Mit->recHit();
	BasicTrajectorySeed::recHitContainer rhContainer;
	if (theCopyMuonRecHit){
	  LogDebug(theCategory)<<"copying ("<<muon.recHitsSize()<<") muon recHits";
	  //copy the muon rechit into the seed
	  for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
	    rhContainer.push_back( (*trit).get()->clone() );  }}
	
	if ( hit->isValid()) {
	  TrajectoryStateOnSurface upState(theUpdator->update(predState,*hit));
	  
	  PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(upState,gd->geographicalId().rawId());
	  LogDebug(theCategory)<<"state used to build a trajectory seed: \n"<<upState
			     <<"on detector: "<<gd->geographicalId().rawId();
	  //add the tracking rechit
	  if (theCopyMuonRecHit){
	    edm::LogError(theCategory)<<"not a bare seed and muon hits are copied. dumping the muon hits.";
	    rhContainer.clear();}
	  rhContainer.push_back(hit->hit()->clone());

	  result.push_back(TrajectorySeed(PTSOD,rhContainer,direction));	  
	}
	else {
	  //rec hit is not valid. put a bare TrajectorySeed, only once !
	  if (!aBareTS){
	    aBareTS=true;
	    
	    PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(predState,gd->geographicalId().rawId());
	    LogDebug(theCategory)<<"state used to build a bare trajectory seed: \n"<<predState
			       <<"on detector: "<<gd->geographicalId().rawId();
    
	    result.push_back(TrajectorySeed(PTSOD,rhContainer,direction));
	  }
	}

      }


    }
  }
  else{
    //transform it into a PTrajectoryStateOnDet
    TrajectoryStateTransform tsTransform;
    PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(compatible.front().second,compatible.front().first->geographicalId().rawId());
    LogDebug(theCategory)<<"state used to build a bare trajectory seed: \n"<<compatible.front().second
		       <<"on detector: "<<compatible.front().first->geographicalId().rawId();
    
    BasicTrajectorySeed::recHitContainer rhContainer;
    if (theCopyMuonRecHit){
      LogDebug(theCategory)<<"copying ("<<muon.recHitsSize()<<") muon recHits";
      //copy the muon rechit into the seed
      for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
	rhContainer.push_back( (*trit).get()->clone() );  }}
    
    //add this seed to the list and return it
    result.push_back(TrajectorySeed(PTSOD,rhContainer,direction));
  }
    return;
}
