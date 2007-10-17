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

TSGForRoadSearch::TSGForRoadSearch(const edm::ParameterSet & par){

  _option = par.getParameter<uint>("option");
  _copyMuonRecHit = par.getParameter<bool>("copyMuonRecHit");
  
  double Chi2 = par.getParameter<double>("maxChi2");
  if (Chi2>0)
    _chi2Estimator = new Chi2MeasurementEstimator(Chi2,sqrt(Chi2));
  else 
    _chi2Estimator=NULL;
  
  _propagatorName = par.getParameter<std::string>("propagatorName");
  _propagatorCompatibleName = par.getParameter<std::string>("propagatorCompatibleName");

  _category = "TSGForRoadSearch|TrackerSeedGenerator";
}
TSGForRoadSearch::~TSGForRoadSearch(){
  delete _chi2Estimator;
}


void TSGForRoadSearch::init(const MuonServiceProxy* service){
  theProxyService = service;
}

void TSGForRoadSearch::setEvent(const edm::Event &event){
  //get the measurementtracker
  theProxyService->eventSetup().get<CkfComponentsRecord>().get(_measurementTracker);
  if (!_measurementTracker.isValid())/*abort*/{edm::LogError("::setEvent()")<<"measurement tracker geometry not found ";}

  //get the global tracking geometry... useless, but hey, what can I do...
  //  iSetup.get<GlobalTrackingGeometryRecord>().get(_glbtrackergeo);
  //  if (!_glbtrackergeo.isValid())/*abort*/{edm::LogError("::setEvent()")<<"global tracking geometry not found";}

  //get the magnetic field
  // service->eventSetup().get<IdealMagneticFieldRecord>().get(_field);
  // if (!_field.isValid())/*abort*/{edm::LogError("::setEvent()")<<"magnetic field not found";}
  
  //get the propagator
  // theProxyService->eventSetup().get<TrackingComponentsRecord>().get(_propagatorName,_prop);
  // if (!_prop.isValid())/*abort*/{edm::LogError("::setEvent()")<<"propagator ("<<_propagatorName<<"_ not found";}

  //get another propagator
  // theProxyService->eventSetup().get<TrackingComponentsRecord>().get(_propagatorCompatibleName,_propCompatible);
  // if (!_propCompatible.isValid())/*abort*/{edm::LogError("::setEvent()")<<"propagator ("<<_propagatorCompatibleName<<"_ not found";}
}


std::vector<TrajectorySeed> TSGForRoadSearch::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region){

  //default result
  std::vector<TrajectorySeed> result;

  switch (_option){
  case 0:
    makeSeeds_0(*muonTrackCand.second,result);break;
  case 1:
    makeSeeds_1(*muonTrackCand.second,result);break;
  case 2:
    makeSeeds_2(*muonTrackCand.second,result);break;
  case 3:
    makeSeeds_3(*muonTrackCand.second,result);break;
  }  

  return result;
}



void TSGForRoadSearch::makeSeeds_2(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  /*
  //define a seed on the outer layers, state is on outer layer and direction is oppositeToMomentum
  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(_category)<<cIPFTS;
  if (cIPFTS.position().mag()==0)  { edm::LogError(_category)<<"initial point of muon is (0,0,0)."; return;} 


  //what are the first layers uncountered
  StartingLayerFinder _finder(theProxyService->propagator(_propagatorName),_measurementTracker.product());
  std::vector<StartingLayerFinder::LayerWithState> layers = _finder.startingOuterStripLayerWithStates(cIPFTS);
  LogDebug(_category)<<"("<<layers.size()<<") starting layers found";

  std::vector< DetLayer::DetWithState > compatible;
  //loop over them to find the first compatible detector
  for (std::vector<StartingLayerFinder::LayerWithState>::iterator itLWS=layers.begin(); itLWS!=layers.end();++itLWS){
    compatible=itLWS->first->compatibleDets(itLWS->second,*theProxyService->propagator(_propagatorCompatibleName),*_chi2Estimator);
    if (!compatible.empty()) break;}
  if (compatible.empty()) {LogDebug(_category)<<"no compatible hits."; return;}
  LogDebug(_category)<<"("<<compatible.size()<<") compatible dets found";

  //transform it into a PTrajectoryStateOnDet
  TrajectoryStateTransform tsTransform;
  PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(compatible.front().second,compatible.front().first->geographicalId().rawId());
  LogDebug(_category)<<"state used to build a trajectory seed: \n"<<compatible.front().second
		     <<"on detector: "<<compatible.front().first->geographicalId().rawId();

  BasicTrajectorySeed::recHitContainer rhContainer;
  if (_copyMuonRecHit){
    LogDebug(_category)<<"copying ("<<muon.recHitsSize()<<") muon recHits";
    //copy the muon rechit into the seed
    for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
      rhContainer.push_back( (*trit).get()->clone() ); }}

  //add this seed to the list and return it
  result.push_back(TrajectorySeed(PTSOD,rhContainer,oppositeToMomentum));
*/
  return;
}

void TSGForRoadSearch::makeSeeds_1(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  /*
  //define a seed on the inner layers, state is on inner layer and direction is alongMomentum

  LogDebug(_category)<<"initial state";

  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(_category)<<cIPFTS;
  if (cIPFTS.position().mag()==0) { edm::LogError(_category)<<"initial point of muon is (0,0,0)."; return;}

  //what are the first layers uncountered
  StartingLayerFinder _finder(theProxyService->propagator(_propagatorName),_measurementTracker.product());
  std::vector<StartingLayerFinder::LayerWithState> layers = _finder.startingStripLayerWithStates(cIPFTS);
  LogDebug(_category)<<"("<<layers.size()<<") starting layers found";

  std::vector< DetLayer::DetWithState > compatible;
  //loop over them to find the first compatible detector
  for (std::vector<StartingLayerFinder::LayerWithState>::iterator itLWS=layers.begin(); itLWS!=layers.end();++itLWS){
    compatible=itLWS->first->compatibleDets(itLWS->second,*theProxyService->propagator(_propagatorCompatibleName),*_chi2Estimator);
    if (!compatible.empty()) break;}
  if (compatible.empty()) {LogDebug(_category)<<"no compatible hits."; return;}
  LogDebug(_category)<<"("<<compatible.size()<<") compatible dets found";


  //transform it into a PTrajectoryStateOnDet
  TrajectoryStateTransform tsTransform;
  PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(compatible.front().second,compatible.front().first->geographicalId().rawId());
  LogDebug(_category)<<"state used to build a trajectory seed: \n"<<compatible.front().second
		     <<"on detector: "<<compatible.front().first->geographicalId().rawId();

  BasicTrajectorySeed::recHitContainer rhContainer;
  if (_copyMuonRecHit){
    LogDebug(_category)<<"copying ("<<muon.recHitsSize()<<") muon recHits";
    //copy the muon rechit into the seed
    for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
      rhContainer.push_back( (*trit).get()->clone() ); }}

  //add this seed to the list and return it
  result.push_back(TrajectorySeed(PTSOD,rhContainer,alongMomentum));
*/  
  return;
}


void TSGForRoadSearch::makeSeeds_0(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(_category)<<cIPFTS;   

  //take state at inner surface and check the first part reached
  std::vector<BarrelDetLayer*> blc = _measurementTracker->geometricSearchTracker()->tibLayers();
  TrajectoryStateOnSurface inner = theProxyService->propagator(_propagatorName)->propagate(cIPFTS,blc.front()->surface());
  if ( !inner.isValid() ) {LogDebug(_category) <<"inner state is not valid"; return;}

  double z = inner.globalPosition().z();

  std::vector<ForwardDetLayer*> ptidc = _measurementTracker->geometricSearchTracker()->posTidLayers();
  std::vector<ForwardDetLayer*> ptecc = _measurementTracker->geometricSearchTracker()->posTecLayers();
  std::vector<ForwardDetLayer*> ntidc = _measurementTracker->geometricSearchTracker()->negTidLayers();
  std::vector<ForwardDetLayer*> ntecc = _measurementTracker->geometricSearchTracker()->negTecLayers();

  const DetLayer *inLayer = NULL;
  if( fabs(z) < ptidc.front()->surface().position().z()  ) {
    inLayer = blc.front();
  } else if ( fabs(z) < ptecc.front()->surface().position().z() ) {
    inLayer = ( z < 0 ) ? ntidc.front() : ptidc.front() ;
  } else {
    inLayer = ( z < 0 ) ? ntecc.front() : ptecc.front() ;
  }

  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible = inLayer->compatibleDets(inner,*theProxyService->propagator(_propagatorCompatibleName),*_chi2Estimator);

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
    compatible = inLayer->compatibleDets(inner,*theProxyService->propagator(_propagatorCompatibleName),*_chi2Estimator);
  }

  //transform it into a PTrajectoryStateOnDet
  TrajectoryStateTransform tsTransform;
  PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(compatible.front().second,compatible.front().first->geographicalId().rawId());
  LogDebug(_category)<<"state used to build a trajectory seed: \n"<<compatible.front().second
		     <<"on detector: "<<compatible.front().first->geographicalId().rawId();

  BasicTrajectorySeed::recHitContainer rhContainer;
  if (_copyMuonRecHit){
    LogDebug(_category)<<"copying ("<<muon.recHitsSize()<<") muon recHits";
    //copy the muon rechit into the seed
    for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
      rhContainer.push_back( (*trit).get()->clone() );  }}

  //add this seed to the list and return it
  result.push_back(TrajectorySeed(PTSOD,rhContainer,alongMomentum));

  return;
}


void TSGForRoadSearch::makeSeeds_3(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  TrajectoryStateTransform transform;
  FreeTrajectoryState cIPFTS = transform.initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(_category)<<cIPFTS;   
  if (cIPFTS.position().mag()==0) /*error*/ { edm::LogError(_category)<<"initial point of muon is (0,0,0)."; return;}

  //take state at inner surface and check the first part reached
  std::vector<BarrelDetLayer*> blc = _measurementTracker->geometricSearchTracker()->tobLayers();
  TrajectoryStateOnSurface outer = theProxyService->propagator(_propagatorName)->propagate(cIPFTS,blc.back()->surface());
  if ( !outer.isValid() ) {LogDebug(_category) <<"outer state is not valid"; return;}

  double z = outer.globalPosition().z();

  std::vector<ForwardDetLayer*> ptidc = _measurementTracker->geometricSearchTracker()->posTidLayers();
  std::vector<ForwardDetLayer*> ptecc = _measurementTracker->geometricSearchTracker()->posTecLayers();
  std::vector<ForwardDetLayer*> ntidc = _measurementTracker->geometricSearchTracker()->negTidLayers();
  std::vector<ForwardDetLayer*> ntecc = _measurementTracker->geometricSearchTracker()->negTecLayers();

  uint layerShift=3;
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
  std::vector< DetLayer::DetWithState > compatible = inLayer->compatibleDets(outer,*theProxyService->propagator(_propagatorCompatibleName),*_chi2Estimator);

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
    compatible = inLayer->compatibleDets(outer,*theProxyService->propagator(_propagatorCompatibleName),*_chi2Estimator);
  }

  //transform it into a PTrajectoryStateOnDet
  TrajectoryStateTransform tsTransform;
  PTrajectoryStateOnDet & PTSOD = *tsTransform.persistentState(compatible.front().second,compatible.front().first->geographicalId().rawId());
  LogDebug(_category)<<"state used to build a trajectory seed: \n"<<compatible.front().second
		     <<"on detector: "<<compatible.front().first->geographicalId().rawId();

  BasicTrajectorySeed::recHitContainer rhContainer;
  if (_copyMuonRecHit){
    LogDebug(_category)<<"copying ("<<muon.recHitsSize()<<") muon recHits";
    //copy the muon rechit into the seed
    for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
      rhContainer.push_back( (*trit).get()->clone() );  }}

  //add this seed to the list and return it
  result.push_back(TrajectorySeed(PTSOD,rhContainer,oppositeToMomentum));

  return;
}
