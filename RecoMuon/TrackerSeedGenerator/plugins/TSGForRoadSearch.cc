#include "RecoMuon/TrackerSeedGenerator/plugins/TSGForRoadSearch.h"

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
#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <TrackingTools/KalmanUpdators/interface/KFUpdator.h>
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"

TSGForRoadSearch::TSGForRoadSearch(const edm::ParameterSet & par,edm::ConsumesCollector& iC){

  theOption = par.getParameter<unsigned int>("option");
  theCopyMuonRecHit = par.getParameter<bool>("copyMuonRecHit");

  double Chi2 = par.getParameter<double>("maxChi2");
  theChi2Estimator = new Chi2MeasurementEstimator(Chi2,sqrt(Chi2));
  
  thePropagatorName = par.getParameter<std::string>("propagatorName");
  thePropagatorCompatibleName = par.getParameter<std::string>("propagatorCompatibleName");

  theCategory = "TSGForRoadSearch|TrackerSeedGenerator";

  theManySeeds = par.getParameter<bool>("manySeeds");
  if (theManySeeds){ theUpdator = new KFUpdator();}
  else{  theUpdator=0;}

  edm::ParameterSet errorMatrixPset = par.getParameter<edm::ParameterSet>("errorMatrixPset");
  if (!errorMatrixPset.empty()){
    theAdjustAtIp = errorMatrixPset.getParameter<bool>("atIP");
    theErrorMatrixAdjuster = new MuonErrorMatrix(errorMatrixPset);}
  else {
    theAdjustAtIp =false;
    theErrorMatrixAdjuster=0;}

  theMeasurementTrackerEventTag = par.getParameter<edm::InputTag>("MeasurementTrackerEvent");
  theMeasurementTrackerEvent = 0;

  theMeasurementTrackerEventToken=iC.consumes<MeasurementTrackerEvent>(theMeasurementTrackerEventTag);
}

TSGForRoadSearch::~TSGForRoadSearch(){
  delete theChi2Estimator;
  if (theUpdator)  delete theUpdator;
  if (theErrorMatrixAdjuster) delete theErrorMatrixAdjuster;
}


void TSGForRoadSearch::init(const MuonServiceProxy* service){
  theProxyService = service;
}

void TSGForRoadSearch::setEvent(const edm::Event &event){
  //get the measurementtracker
  if (theManySeeds){
    theProxyService->eventSetup().get<CkfComponentsRecord>().get(theMeasurementTracker);
    if (!theMeasurementTracker.isValid())/*abort*/{edm::LogError(theCategory)<<"measurement tracker geometry not found ";}
  }
  theProxyService->eventSetup().get<TrackerRecoGeometryRecord>().get(theGeometricSearchTracker);

  edm::Handle<MeasurementTrackerEvent> data;
  event.getByToken(theMeasurementTrackerEventToken, data);
  theMeasurementTrackerEvent = &*data;
}



void  TSGForRoadSearch::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region, const TrackerTopology *tTopo,
				     std::vector<TrajectorySeed> & result){
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

bool TSGForRoadSearch::notAtIPtsos(TrajectoryStateOnSurface & state){
  LogDebug(theCategory)<<"outer state: "<<state;
  if (theErrorMatrixAdjuster && !theAdjustAtIp){
    theErrorMatrixAdjuster->adjust(state);
    LogDebug(theCategory)<<"outer state after rescale: "<<state;
  }
  return true;
}

bool TSGForRoadSearch::IPfts(const reco::Track & muon, FreeTrajectoryState & fts){
  
  fts = trajectoryStateTransform::initialFreeState(muon,&*theProxyService->magneticField());
  LogDebug(theCategory)<<"pure L2 state: "<<fts;
  if (fts.position().mag()==0 && fts.momentum().mag()==0){ edm::LogError(theCategory)<<"initial state of muon is (0,0,0)(0,0,0). no seed."; 
    return false;}

  //rescale the error at IP
  if (theErrorMatrixAdjuster && theAdjustAtIp){ theErrorMatrixAdjuster->adjust(fts);
    LogDebug(theCategory)<<"after adjusting the error matrix: "<<fts;}

  return true;
}

//-----------------------------------------
// inside-out generator option NO pixel used
//-----------------------------------------
void TSGForRoadSearch::makeSeeds_0(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  FreeTrajectoryState cIPFTS;
  if (!IPfts(muon, cIPFTS)) return;

  //take state at inner surface and check the first part reached
  const std::vector<BarrelDetLayer*> & blc = theGeometricSearchTracker->tibLayers();
  TrajectoryStateOnSurface inner = theProxyService->propagator(thePropagatorName)->propagate(cIPFTS,blc.front()->surface());
  if ( !inner.isValid() ) {LogDebug(theCategory) <<"inner state is not valid. no seed."; return;}

  //rescale the error
  if (!notAtIPtsos(inner)) return;

  double z = inner.globalPosition().z();

  const std::vector<ForwardDetLayer*> &ptidc = theGeometricSearchTracker->posTidLayers();
  const std::vector<ForwardDetLayer*> &ptecc = theGeometricSearchTracker->posTecLayers();
  const std::vector<ForwardDetLayer*> &ntidc = theGeometricSearchTracker->negTidLayers();
  const std::vector<ForwardDetLayer*> &ntecc = theGeometricSearchTracker->negTecLayers();

  const DetLayer *inLayer = 0;
  if( fabs(z) < ptidc.front()->surface().position().z()  ) {
    inLayer = blc.front();
  } else if ( fabs(z) < ptecc.front()->surface().position().z() ) {
    inLayer = ( z < 0 ) ? ntidc.front() : ptidc.front() ;
  } else {
    inLayer = ( z < 0 ) ? ntecc.front() : ptecc.front() ;
  }

  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible;
  compatible.reserve(10);
  inLayer->compatibleDetsV(inner,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator,compatible);

  //loop the parts until at least a compatible is found
  while (compatible.size()==0) {
    switch ( inLayer->subDetector() ) {
    case GeomDetEnumerators::PixelBarrel:
    case GeomDetEnumerators::PixelEndcap:
    case GeomDetEnumerators::TOB:
    case GeomDetEnumerators::TEC:
      LogDebug(theCategory)<<"from inside-out, trying TEC or TOB layers. no seed.";
      return;
      break;
    case GeomDetEnumerators::TIB:
      inLayer = ( z < 0 ) ? ntidc.front() : ptidc.front() ;
      break;
    case GeomDetEnumerators::TID:
      inLayer = ( z < 0 ) ? ntecc.front() : ptecc.front() ;
      break;
    default:
      LogDebug(theCategory)<<"subdetectorid is not a tracker sub-dectector id. skipping.";
      return;
    }
    inLayer->compatibleDetsV(inner,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator,compatible);
  }

  pushTrajectorySeed(muon,compatible,alongMomentum,result);

  return;
}

void TSGForRoadSearch::makeSeeds_1(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  edm::LogError(theCategory)<<"option 1 of TSGForRoadSearch is not implemented yet. Please use 0,3 or 4. no seed.";
  return;
}

void TSGForRoadSearch::makeSeeds_2(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  edm::LogError(theCategory)<<"option 2 of TSGForRoadSearch is not implemented yet. Please use 0,3 or 4. no seed.";
  return;
}

//---------------------------------
// outside-in seed generator option
//---------------------------------
void TSGForRoadSearch::makeSeeds_3(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  FreeTrajectoryState cIPFTS;
  if (!IPfts(muon, cIPFTS)) return;

  //take state at outer surface and check the first part reached
  const std::vector<BarrelDetLayer*> &blc = theGeometricSearchTracker->tobLayers();

  //  TrajectoryStateOnSurface outer = theProxyService->propagator(thePropagatorName)->propagate(cIPFTS,blc.back()->surface());
  StateOnTrackerBound onBounds(theProxyService->propagator(thePropagatorName).product());
  TrajectoryStateOnSurface outer = onBounds(cIPFTS);

  if ( !outer.isValid() ) {LogDebug(theCategory) <<"outer state is not valid. no seed."; return;}
  
  //rescale the error
  if (!notAtIPtsos(outer)) return;

  double z = outer.globalPosition().z();

  const std::vector<ForwardDetLayer*> &ptecc = theGeometricSearchTracker->posTecLayers();
  const std::vector<ForwardDetLayer*> &ntecc = theGeometricSearchTracker->negTecLayers();

  LogDebug(theCategory)<<"starting looking for a compatible layer from: "<<outer<<"\nz: "<<z<<"TEC1 z: "<<ptecc.front()->surface().position().z();

  unsigned int layerShift=0;
  const DetLayer *inLayer = 0;
  if (fabs(z) < ptecc.front()->surface().position().z()  ){
    inLayer = *(blc.rbegin()+layerShift);
    LogTrace(theCategory)<<"choosing TOB layer with shift: "<<layerShift;
  } else {
    unsigned int tecIt=1;
    for (; tecIt!=ptecc.size();tecIt++){
      LogTrace(theCategory)<<"checking surface with shift: "<<tecIt
			   <<"z: "<<ptecc[tecIt]->surface().position().z();
      if (fabs(z) < ptecc[tecIt]->surface().position().z())
	{inLayer = ( z < 0 ) ? ntecc[tecIt-1] : ptecc[tecIt-1] ; 
	  layerShift=tecIt-1;
	  LogTrace(theCategory)<<"choosing TEC layer with shift: "<<layerShift
			       <<" and z: "<<inLayer->surface().position().z();
	  break;}}
    if (!inLayer) {inLayer = ( z < 0 ) ? ntecc.back() : ptecc.back();
      LogTrace(theCategory)<<"choosing last TEC layer with z: "<<inLayer->surface().position().z();
    }
  }

  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible;
  compatible.reserve(10);
  inLayer->compatibleDetsV(outer,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator,compatible);

  //loop the parts until at least a compatible is found
  while (compatible.size()==0) {
    switch ( inLayer->subDetector() ) {
    case GeomDetEnumerators::PixelBarrel:
    case GeomDetEnumerators::PixelEndcap:
    case GeomDetEnumerators::TIB:
    case GeomDetEnumerators::TID:
    case GeomDetEnumerators::TOB:
      layerShift++;
      if (layerShift>=blc.size()){
	LogDebug(theCategory) <<"all barrel layers are exhausted to find starting state. no seed,";
	return;}
      inLayer = *(blc.rbegin()+layerShift);
      break;
    case GeomDetEnumerators::TEC:
      if (layerShift==0){
	LogDebug(theCategory) <<"failed to get a compatible module on a TEC layer, using the last TOB layer.";
	inLayer = *(blc.rbegin()+layerShift);
      }
      else{
	layerShift--;
	LogDebug(theCategory) <<"reaching more in with layer "<<layerShift<<" in TEC";
	inLayer = ( z < 0 ) ? ntecc[layerShift] : ptecc[layerShift] ;
      }
      break;
    default:
      edm::LogError(theCategory)<<"subdetectorid is not a tracker sub-dectector id. skipping.";
      return;
    }
    inLayer->compatibleDetsV(outer,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator,compatible);
  }

  pushTrajectorySeed(muon,compatible,oppositeToMomentum,result);

  return;
}


//-----------------------------------------
// inside-out generator option, using pixel
//-----------------------------------------
void TSGForRoadSearch::makeSeeds_4(const reco::Track & muon, std::vector<TrajectorySeed>& result){
  //get the state at IP
  FreeTrajectoryState cIPFTS;
  if (!IPfts(muon, cIPFTS)) return;

  //take state at inner surface and check the first part reached
  const std::vector<BarrelDetLayer*> & blc = theGeometricSearchTracker->pixelBarrelLayers();
  if (blc.empty()){edm::LogError(theCategory)<<"want to start from pixel layer, but no barrel exists. trying without pixel."; 
    makeSeeds_0(muon, result);
    return;}

  TrajectoryStateOnSurface inner = theProxyService->propagator(thePropagatorName)->propagate(cIPFTS,blc.front()->surface());
  if ( !inner.isValid() ) {LogDebug(theCategory) <<"inner state is not valid. no seed."; return;}

  //rescale the error
  if (!notAtIPtsos(inner)) return;
    
  double z = inner.globalPosition().z();

  const std::vector<ForwardDetLayer*> &ppxlc = theGeometricSearchTracker->posPixelForwardLayers();
  const std::vector<ForwardDetLayer*> &npxlc = theGeometricSearchTracker->negPixelForwardLayers();
  const std::vector<ForwardDetLayer*> &ptidc = theGeometricSearchTracker->posTidLayers();
  const std::vector<ForwardDetLayer*> &ptecc = theGeometricSearchTracker->posTecLayers();
  const std::vector<ForwardDetLayer*> &ntidc = theGeometricSearchTracker->negTidLayers();
  const std::vector<ForwardDetLayer*> &ntecc = theGeometricSearchTracker->negTecLayers();

  if ((ppxlc.empty() || npxlc.empty()) && (ptidc.empty() || ptecc.empty()) )
    { edm::LogError(theCategory)<<"want to start from pixel layer, but no forward layer exists. trying without pixel.";
      makeSeeds_0(muon, result);
      return;}

  const DetLayer *inLayer = 0;
  std::vector<ForwardDetLayer*>::const_iterator layerIt ;

  double fz=fabs(z);
  
  //simple way of finding a first layer to try out
  if (fz < fabs(((z>0)?ppxlc:npxlc).front()->surface().position().z())){
    inLayer = blc.front();}
  else if (fz < fabs(((z>0)?ppxlc:npxlc).back()->surface().position().z())){
    layerIt = ((z>0)?ppxlc:npxlc).begin();
    inLayer= *layerIt;}
  else if (fz < fabs(((z>0)?ptidc:ntidc).front()->surface().position().z())){
    layerIt = ((z>0)?ppxlc:npxlc).end()-1;
    inLayer= *layerIt;}
  else if (fz < fabs(((z>0)?ptecc:ntecc).front()->surface().position().z())){
    layerIt = ((z>0)?ptidc:ntidc).begin();
    inLayer= *layerIt;}
  else if (fz < fabs(((z>0)?ptecc:ntecc).back()->surface().position().z())){
    layerIt = ((z>0)?ptecc:ntecc).begin();
    inLayer= *layerIt;}
  else {
    edm::LogWarning(theCategory)<<"the state is not consistent with any tracker layer:\n"
			     <<inner;
    return;}
  
  //find out at least one compatible detector reached
  std::vector< DetLayer::DetWithState > compatible;
  compatible.reserve(10);
  inLayer->compatibleDetsV(inner,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator,compatible);
  
  //if none were found. you should do something more.
  if (compatible.size()==0){
    std::vector<ForwardDetLayer*>::const_iterator pxlEnd = (z>0)? ppxlc.end() : npxlc.end();
    std::vector<ForwardDetLayer*>::const_iterator tidEnd = (z>0)? ptidc.end() : ntidc.end();
    std::vector<ForwardDetLayer*>::const_iterator tecEnd = (z>0)? ptecc.end() : ntecc.end();
    std::vector<ForwardDetLayer*>::const_iterator pxlBegin = (z>0)? ppxlc.begin() : npxlc.begin();
    std::vector<ForwardDetLayer*>::const_iterator tidBegin = (z>0)? ptidc.begin() : ntidc.begin();
    std::vector<ForwardDetLayer*>::const_iterator tecBegin = (z>0)? ptecc.begin() : ntecc.begin();

    //go to first disk if not already in a disk situation
    if (!dynamic_cast<const ForwardDetLayer*>(inLayer)) layerIt =pxlBegin--;
    
    while (compatible.size()==0) {
      switch ( (*layerIt)->subDetector() ) {
      case GeomDetEnumerators::PixelEndcap:
	{
	  layerIt++;
	  //if end of list reached. go to the first TID
	  if (layerIt==pxlEnd) layerIt=tidBegin;
	  break;
	}
      case GeomDetEnumerators::TID:
	{
	  layerIt++;
	  //if end of list reached. go to the first TEC
	  if (layerIt==tidEnd) layerIt = tecBegin;
	  break;
	}
      case GeomDetEnumerators::TEC:
	{
	  layerIt++;
	  if (layerIt==tecEnd){
	    edm::LogWarning(theCategory)<<"ran out of layers to find a seed: no seed.";
	    return;}
	}
      case GeomDetEnumerators::PixelBarrel: { edm::LogError(theCategory)<<"this should not happen... ever. Please report. GeomDetEnumerators::PixelBarrel. no seed."; return;}
      case GeomDetEnumerators::TIB: { edm::LogError(theCategory)<<"this should not happen... ever. Please report. GeomDetEnumerators::TIB. no seed."; return;}
      case GeomDetEnumerators::TOB: { edm::LogError(theCategory)<<"this should not happen... ever. Please report. GeomDetEnumerators::TOB. no seed."; return;}
      default:	{ edm::LogError(theCategory)<<"Subdetector id is not a tracker sub-detector id. no seed."; return;}
      }//switch

      (*layerIt)->compatibleDetsV(inner,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator,compatible);
    }//while
  }//if size==0

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
    

    //finf out every compatible measurements
    for (std::vector<DetLayer::DetWithState >::iterator DWSit = compatible.begin(); DWSit!=compatible.end();++DWSit){
      bool aBareTS=false;
      const GeomDet * gd = DWSit->first;
      if (!gd){edm::LogError(theCategory)<<"GeomDet is not valid."; continue;}
      MeasurementDetWithData md = theMeasurementTrackerEvent->idToDet(gd->geographicalId());
      std::vector<TrajectoryMeasurement> tmp = md.fastMeasurements(DWSit->second,DWSit->second,*theProxyService->propagator(thePropagatorCompatibleName),*theChi2Estimator);
      //make a trajectory seed for each of them

      for (std::vector<TrajectoryMeasurement>::iterator Mit = tmp.begin(); Mit!=tmp.end();++Mit){
	TrajectoryStateOnSurface predState(Mit->predictedState());
	TrajectoryMeasurement::ConstRecHitPointer hit = Mit->recHit();
	TrajectorySeed::recHitContainer rhContainer;
	if (theCopyMuonRecHit){
	  LogDebug(theCategory)<<"copying ("<<muon.recHitsSize()<<") muon recHits";
	  //copy the muon rechit into the seed
	  for (trackingRecHit_iterator trit = muon.recHitsBegin(); trit!=muon.recHitsEnd();trit++) {
	    rhContainer.push_back( (*trit).get()->clone() );  }}
	
	if ( hit->isValid()) {
	  TrajectoryStateOnSurface upState(theUpdator->update(predState,*hit));
	  
	  PTrajectoryStateOnDet const & PTSOD = trajectoryStateTransform::persistentState(upState,gd->geographicalId().rawId());
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
	    
	    PTrajectoryStateOnDet const & PTSOD = trajectoryStateTransform::persistentState(predState,gd->geographicalId().rawId());
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
    
    PTrajectoryStateOnDet const& PTSOD = trajectoryStateTransform::persistentState(compatible.front().second,compatible.front().first->geographicalId().rawId());
    LogDebug(theCategory)<<"state used to build a bare trajectory seed: \n"<<compatible.front().second
		       <<"on detector: "<<compatible.front().first->geographicalId().rawId();
    
    TrajectorySeed::recHitContainer rhContainer;
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
