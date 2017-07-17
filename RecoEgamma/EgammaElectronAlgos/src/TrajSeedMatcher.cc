#include "RecoEgamma/EgammaElectronAlgos/interface/TrajSeedMatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h" 

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

constexpr float TrajSeedMatcher::kElectronMass_;

TrajSeedMatcher::TrajSeedMatcher(const edm::ParameterSet& pset):
  cacheIDMagField_(0),
  minNrHits_(pset.getParameter<std::vector<unsigned int> >("minNrHits")),
  minNrHitsValidLayerBins_(pset.getParameter<std::vector<int> >("minNrHitsValidLayerBins"))
{
  useRecoVertex_ = pset.getParameter<bool>("useRecoVertex");
  navSchoolLabel_ = pset.getParameter<std::string>("navSchool");
  detLayerGeomLabel_ = pset.getParameter<std::string>("detLayerGeom");
  const auto cutsPSets=pset.getParameter<std::vector<edm::ParameterSet> >("matchingCuts");
  for(const auto & cutPSet : cutsPSets){
    int version=cutPSet.getParameter<int>("version");
    switch(version){
    case 1:
      matchingCuts_.emplace_back(std::make_unique<MatchingCutsV1>(cutPSet));
      break;
    case 2:
      matchingCuts_.emplace_back(std::make_unique<MatchingCutsV2>(cutPSet));
      break;
    default:
      throw cms::Exception("InvalidConfig") <<" Error TrajSeedMatcher::TrajSeedMatcher pixel match cuts version "<<version<<" not recognised"<<std::endl;
    }
  }
 
  if(minNrHitsValidLayerBins_.size()+1!=minNrHits_.size()){  
    throw cms::Exception("InvalidConfig")<<" TrajSeedMatcher::TrajSeedMatcher minNrHitsValidLayerBins should be 1 less than minNrHits when its "<<minNrHitsValidLayerBins_.size()<<" vs "<<minNrHits_.size();
  }
}

edm::ParameterSetDescription TrajSeedMatcher::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.add<bool>("useRecoVertex",false);
  desc.add<std::string>("navSchool","SimpleNavigationSchool");
  desc.add<std::string>("detLayerGeom","hltESPGlobalDetLayerGeometry");
  desc.add<std::vector<int> >("minNrHitsValidLayerBins",{4});
  desc.add<std::vector<unsigned int> >("minNrHits",{2,3});
  
  edm::ParameterSetDescription cutsDesc;
  auto cutDescCases = 
    1 >> 
    (edm::ParameterDescription<double>("dPhiMax",0.04,true) and
     edm::ParameterDescription<double>("dRZMax",0.09,true) and
     edm::ParameterDescription<double>("dRZMaxLowEtThres",20.,true) and
     edm::ParameterDescription<std::vector<double> >("dRZMaxLowEtEtaBins",std::vector<double>{1.,1.5},true) and
     edm::ParameterDescription<std::vector<double> >("dRZMaxLowEt",std::vector<double>{0.09,0.15,0.09},true)) or
    2 >> 
    (edm::ParameterDescription<std::vector<double> >("dPhiMaxHighEt",{0.003},true) and
     edm::ParameterDescription<std::vector<double> >("dPhiMaxHighEtThres",{0.0},true) and
     edm::ParameterDescription<std::vector<double> >("dPhiMaxLowEtGrad",{0.0},true) and
     edm::ParameterDescription<std::vector<double> >("dRZMaxHighEt",{0.005},true) and
     edm::ParameterDescription<std::vector<double> >("dRZMaxHighEtThres",{30},true) and
     edm::ParameterDescription<std::vector<double> >("dRZMaxLowEtGrad",{-0.002},true) and
     edm::ParameterDescription<std::vector<double> >("etaBins",{},true));
  cutsDesc.ifValue(edm::ParameterDescription<int>("version",1,true), std::move(cutDescCases));

  edm::ParameterSet defaults;
  defaults.addParameter<double>("dPhiMax",0.04);
  defaults.addParameter<double>("dRZMax",0.09);
  defaults.addParameter<double>("dRZMaxLowEtThres",0.09);
  defaults.addParameter<std::vector<double> >("dRZMaxLowEtEtaBins",std::vector<double>{1.,1.5});
  defaults.addParameter<std::vector<double> >("dRZMaxLowEt",std::vector<double>{0.09,0.09,0.09});
  defaults.addParameter<int>("version",1);
  desc.addVPSet("matchingCuts",cutsDesc,std::vector<edm::ParameterSet>{defaults,defaults,defaults});
  return desc;
}

void TrajSeedMatcher::doEventSetup(const edm::EventSetup& iSetup)
{
  if (cacheIDMagField_!=iSetup.get<IdealMagneticFieldRecord>().cacheIdentifier()) {
    iSetup.get<IdealMagneticFieldRecord>().get(magField_);
    cacheIDMagField_=iSetup.get<IdealMagneticFieldRecord>().cacheIdentifier();
    forwardPropagator_=std::make_unique<PropagatorWithMaterial>(alongMomentum,kElectronMass_,&*(magField_));
    backwardPropagator_=std::make_unique<PropagatorWithMaterial>(oppositeToMomentum,kElectronMass_,&*(magField_));
  }
  iSetup.get<NavigationSchoolRecord>().get(navSchoolLabel_,navSchool_);
  iSetup.get<RecoGeometryRecord>().get(detLayerGeomLabel_,detLayerGeom_);
}


std::vector<TrajSeedMatcher::SeedWithInfo>
TrajSeedMatcher::compatibleSeeds(const TrajectorySeedCollection& seeds, const GlobalPoint& candPos,
				  const GlobalPoint & vprim, const float energy)
{
  if(!forwardPropagator_ || !backwardPropagator_ || !magField_.isValid()){
    throw cms::Exception("LogicError") <<__FUNCTION__<<" can not make pixel seeds as event setup has not properly been called";
  }

  clearCache();
  
  std::vector<SeedWithInfo> matchedSeeds;
  for(const auto& seed : seeds) {
    std::vector<SCHitMatch> matchedHitsNeg = processSeed(seed,candPos,vprim,energy,-1);
    std::vector<SCHitMatch> matchedHitsPos = processSeed(seed,candPos,vprim,energy,+1);
    int nrValidLayersPos = 0;
    int nrValidLayersNeg = 0;
    if(matchedHitsNeg.size()>=2){
      nrValidLayersNeg = getNrValidLayersAlongTraj(matchedHitsNeg[0],
						   matchedHitsNeg[1],
						   candPos,vprim,energy,-1);
    }
    if(matchedHitsPos.size()>=2){
      nrValidLayersPos = getNrValidLayersAlongTraj(matchedHitsPos[0],
						   matchedHitsPos[1],
						   candPos,vprim,energy,+1);
    }
    
    int nrValidLayers = std::max(nrValidLayersNeg,nrValidLayersPos);
    size_t nrHitsRequired = getNrHitsRequired(nrValidLayers);
    //so we require the number of hits to exactly match, this is because we currently do not
    //do any duplicate cleaning for the input seeds
    //this means is a hit pair with a 3rd hit will appear twice (or however many hits it has)
    //so if you did >=nrHitsRequired, you would get the same seed multiple times
    //ideally we should fix this and clean our input seed collection so each seed is only in once
    //also it should be studied what impact having a 3rd hit has on a GsfTrack
    //ie will we get a significantly different result seeding with a hit pair 
    //and the same the hit pair with a 3rd hit added
    //in that case, perhaps it should be >=
    if(matchedHitsNeg.size()==nrHitsRequired ||
       matchedHitsPos.size()==nrHitsRequired){
      matchedSeeds.push_back({seed,matchedHitsPos,matchedHitsNeg,nrValidLayers});
    }
    

  }
  return matchedSeeds;
}

//checks if the hits of the seed match within requested selection
//matched hits are required to be consecutive, as soon as hit isnt matched,
//the function returns, it doesnt allow skipping hits
std::vector<TrajSeedMatcher::SCHitMatch>
TrajSeedMatcher::processSeed(const TrajectorySeed& seed, const GlobalPoint& candPos,
			     const GlobalPoint & vprim, const float energy, const int charge )
{
  const float candEta = candPos.eta();
  const float candEt = energy*std::sin(candPos.theta());
  
  FreeTrajectoryState trajStateFromVtx = FTSFromVertexToPointFactory::get(*magField_, candPos, vprim, energy, charge);
  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface initialTrajState(trajStateFromVtx,*bpb(trajStateFromVtx.position(), 
								  trajStateFromVtx.momentum()));
 
  std::vector<SCHitMatch> matchedHits;
  SCHitMatch firstHit = matchFirstHit(seed,initialTrajState,vprim,*backwardPropagator_);
  firstHit.setExtra(candEt,candEta,candPos.phi(),charge,1);
  if(passesMatchSel(firstHit,0)){
    matchedHits.push_back(firstHit);

    //now we can figure out the z vertex
    double zVertex = useRecoVertex_ ? vprim.z() : getZVtxFromExtrapolation(vprim,firstHit.hitPos(),candPos);
    GlobalPoint vertex(vprim.x(),vprim.y(),zVertex);
    
    FreeTrajectoryState firstHitFreeTraj = FTSFromVertexToPointFactory::get(*magField_, firstHit.hitPos(), 
									    vertex, energy, charge) ;
 
    GlobalPoint prevHitPos = firstHit.hitPos();
    for(size_t hitNr=1;hitNr<matchingCuts_.size() && hitNr<seed.nHits();hitNr++){
      SCHitMatch hit = match2ndToNthHit(seed,firstHitFreeTraj,hitNr,prevHitPos,vertex,*forwardPropagator_);
      hit.setExtra(candEt,candEta,candPos.phi(),charge,1);
      if(passesMatchSel(hit,hitNr)){
	matchedHits.push_back(hit);
	prevHitPos = hit.hitPos();
      }else break;
    }
  }
  return matchedHits;
}

// compute the z vertex from the candidate position and the found pixel hit
float TrajSeedMatcher::getZVtxFromExtrapolation(const GlobalPoint& primeVtxPos, const GlobalPoint& hitPos,
						const GlobalPoint& candPos)
{
  auto sq = [](float x){return x*x;};
  auto calRDiff = [sq](const GlobalPoint& p1,const GlobalPoint& p2){
    return std::sqrt(sq(p2.x()-p1.x()) + sq(p2.y()-p1.y()));
  };
  const double r1Diff = calRDiff(primeVtxPos,hitPos);
  const double r2Diff = calRDiff(hitPos,candPos);
  return hitPos.z() - r1Diff*(candPos.z()-hitPos.z())/r2Diff;
}

bool TrajSeedMatcher::passTrajPreSel(const GlobalPoint& hitPos, const GlobalPoint& candPos)const
{
  float dt = hitPos.x()*candPos.x()+hitPos.y()*candPos.y();
  if (dt<0) return false;
  if (dt<kPhiCut_*(candPos.perp()*hitPos.perp())) return false;
  return true;
}

const TrajectoryStateOnSurface& TrajSeedMatcher::getTrajStateFromVtx(const TrackingRecHit& hit,
								     const TrajectoryStateOnSurface& initialState,
								     const PropagatorWithMaterial& propagator)
{
  auto& trajStateFromVtxCache = initialState.charge()==1 ? trajStateFromVtxPosChargeCache_ :
                                                           trajStateFromVtxNegChargeCache_;

  auto key = hit.det()->gdetIndex();
  auto res = trajStateFromVtxCache.find(key);
  if(res!=trajStateFromVtxCache.end()) return res->second;
  else{ //doesnt exist, need to make it
    //FIXME: check for efficiency
    auto val = trajStateFromVtxCache.emplace(key,propagator.propagate(initialState,hit.det()->surface()));
    return val.first->second;
  }
}

const TrajectoryStateOnSurface& TrajSeedMatcher::getTrajStateFromPoint(const TrackingRecHit& hit,
								       const FreeTrajectoryState& initialState,
								       const GlobalPoint& point,
								       const PropagatorWithMaterial& propagator)
{
  
  auto& trajStateFromPointCache = initialState.charge()==1 ? trajStateFromPointPosChargeCache_ :
                                                             trajStateFromPointNegChargeCache_;

  auto key = std::make_pair(hit.det()->gdetIndex(),point);
  auto res = trajStateFromPointCache.find(key);
  if(res!=trajStateFromPointCache.end()) return res->second;
  else{ //doesnt exist, need to make it
    //FIXME: check for efficiency
    auto val = trajStateFromPointCache.emplace(key,propagator.propagate(initialState,hit.det()->surface()));
    return val.first->second;
  }
}

TrajSeedMatcher::SCHitMatch TrajSeedMatcher::matchFirstHit(const TrajectorySeed& seed,
							   const TrajectoryStateOnSurface& initialState,
							   const GlobalPoint& vtxPos,
							   const PropagatorWithMaterial& propagator)
{
  const TrajectorySeed::range& hits = seed.recHits();
  auto hitIt = hits.first;

  if(hitIt->isValid()){
    const TrajectoryStateOnSurface& trajStateFromVtx = getTrajStateFromVtx(*hitIt,initialState,propagator);
    if(trajStateFromVtx.isValid()) return SCHitMatch(vtxPos,trajStateFromVtx,*hitIt);  
  }
  return SCHitMatch();
}

TrajSeedMatcher::SCHitMatch TrajSeedMatcher::match2ndToNthHit(const TrajectorySeed& seed,
							   const FreeTrajectoryState& initialState,
							   const size_t hitNr,
							   const GlobalPoint& prevHitPos,
							   const GlobalPoint& vtxPos,
							   const PropagatorWithMaterial& propagator)
{
  const TrajectorySeed::range& hits = seed.recHits();
  auto hitIt = hits.first+hitNr;
  
  if(hitIt->isValid()){
    const TrajectoryStateOnSurface& trajState = getTrajStateFromPoint(*hitIt,initialState,prevHitPos,propagator);
    
    if(trajState.isValid()){
      return SCHitMatch(vtxPos,trajState,*hitIt);  
    }
  }
  return SCHitMatch();
  
}

void TrajSeedMatcher::clearCache()
{
  trajStateFromVtxPosChargeCache_.clear();
  trajStateFromVtxNegChargeCache_.clear();
  trajStateFromPointPosChargeCache_.clear();
  trajStateFromPointNegChargeCache_.clear();
}

bool TrajSeedMatcher::passesMatchSel(const TrajSeedMatcher::SCHitMatch& hit, const size_t hitNr)const
{
  if(hitNr<matchingCuts_.size()){
    return (*matchingCuts_[hitNr])(hit);
  }else{
    throw cms::Exception("LogicError") <<" Error, attempting to apply selection to hit "<<hitNr<<" but only cuts for "<<matchingCuts_.size()<<" defined";
  }
}

int TrajSeedMatcher::getNrValidLayersAlongTraj(const SCHitMatch& hit1, const SCHitMatch& hit2,
						const GlobalPoint& candPos,
						const GlobalPoint & vprim, 
						const float energy, const int charge)
{
  double zVertex = useRecoVertex_ ? vprim.z() : getZVtxFromExtrapolation(vprim,hit1.hitPos(),candPos);
  GlobalPoint vertex(vprim.x(),vprim.y(),zVertex);
  
  FreeTrajectoryState firstHitFreeTraj = FTSFromVertexToPointFactory::get(*magField_,hit1.hitPos(), 
									  vertex, energy, charge);
  const TrajectoryStateOnSurface& secondHitTraj = getTrajStateFromPoint(*hit2.hit(),firstHitFreeTraj,hit1.hitPos(),*forwardPropagator_);
  return getNrValidLayersAlongTraj(hit2.hit()->geographicalId(),secondHitTraj); 
}

int TrajSeedMatcher::getNrValidLayersAlongTraj(const DetId& hitId, const TrajectoryStateOnSurface& hitTrajState)const
{
  
  const DetLayer* detLayer = detLayerGeom_->idToLayer(hitId);
  if(detLayer==nullptr) return 0;

  const FreeTrajectoryState& hitFreeState = *hitTrajState.freeState();
  const std::vector<const DetLayer*> inLayers  = navSchool_->compatibleLayers(*detLayer,hitFreeState,oppositeToMomentum); 
  const std::vector<const DetLayer*> outLayers = navSchool_->compatibleLayers(*detLayer,hitFreeState,alongMomentum); 
  
  int nrValidLayers=1; //because our current hit is also valid and wont be included in the count otherwise
  int nrPixInLayers=0;
  int nrPixOutLayers=0;
  for(auto layer : inLayers){
    if(GeomDetEnumerators::isTrackerPixel(layer->subDetector())){ 
      nrPixInLayers++;
      if(layerHasValidHits(*layer,hitTrajState,*backwardPropagator_)) nrValidLayers++;  
    }
  }
  for(auto layer : outLayers){
    if(GeomDetEnumerators::isTrackerPixel(layer->subDetector())){ 
      nrPixOutLayers++;
      if(layerHasValidHits(*layer,hitTrajState,*forwardPropagator_)) nrValidLayers++;  
    }
  }
  return nrValidLayers;
}
						 
bool TrajSeedMatcher::layerHasValidHits(const DetLayer& layer, const TrajectoryStateOnSurface& hitSurState,
					const Propagator& propToLayerFromState)const
{
  //FIXME: do not hardcode with werid magic numbers stolen from ancient tracking code
  //its taken from https://cmssdt.cern.ch/dxr/CMSSW/source/RecoTracker/TrackProducer/interface/TrackProducerBase.icc#165
  //which inspires this code
  Chi2MeasurementEstimator estimator(30.,-3.0,0.5,2.0,0.5,1.e12);  // same as defauts....
  
  const std::vector<GeometricSearchDet::DetWithState>& detWithState = layer.compatibleDets(hitSurState,propToLayerFromState,estimator);
  if(detWithState.empty()) return false;
  else{
    DetId id = detWithState.front().first->geographicalId();
    MeasurementDetWithData measDet = measTkEvt_->idToDet(id);
    if(measDet.isActive()) return true;
    else return false;
  }
}


size_t TrajSeedMatcher::getNrHitsRequired(const int nrValidLayers)const
{
  for(size_t binNr=0;binNr<minNrHitsValidLayerBins_.size();binNr++){
    if(nrValidLayers<minNrHitsValidLayerBins_[binNr]) return minNrHits_[binNr];
  }
  return minNrHits_.back();
  
}

TrajSeedMatcher::SCHitMatch::SCHitMatch(const GlobalPoint& vtxPos,
					const TrajectoryStateOnSurface& trajState,
					const TrackingRecHit& hit):
  detId_(hit.geographicalId()),
  hitPos_(hit.globalPosition()),
  hit_(&hit),
  et_(0),eta_(0),phi_(0),charge_(0),nrClus_(0)
{
  EleRelPointPair pointPair(hitPos_,trajState.globalParameters().position(),vtxPos);
  dRZ_ = detId_.subdetId()==PixelSubdetector::PixelBarrel ? pointPair.dZ() : pointPair.dPerp();
  dPhi_ = pointPair.dPhi();
}
    

TrajSeedMatcher::SeedWithInfo::
SeedWithInfo(const TrajectorySeed& seed,
	     const std::vector<SCHitMatch>& posCharge,
	     const std::vector<SCHitMatch>& negCharge,
	     int nrValidLayers):
  seed_(seed),nrValidLayers_(nrValidLayers)
{
  size_t nrHitsMax = std::max(posCharge.size(),negCharge.size());
  for(size_t hitNr=0;hitNr<nrHitsMax;hitNr++){
    DetId detIdPos = hitNr<posCharge.size() ? posCharge[hitNr].detId() : DetId(0);
    float dRZPos = hitNr<posCharge.size() ? posCharge[hitNr].dRZ() : std::numeric_limits<float>::max();
    float dPhiPos = hitNr<posCharge.size() ? posCharge[hitNr].dPhi() : std::numeric_limits<float>::max();

    DetId detIdNeg = hitNr<negCharge.size() ? negCharge[hitNr].detId() : DetId(0);
    float dRZNeg = hitNr<negCharge.size() ? negCharge[hitNr].dRZ() : std::numeric_limits<float>::max();
    float dPhiNeg = hitNr<negCharge.size() ? negCharge[hitNr].dPhi() : std::numeric_limits<float>::max();
    
    if(detIdPos!=detIdNeg && (detIdPos.rawId()!=0 && detIdNeg.rawId()!=0)){
      cms::Exception("LogicError")<<" error in "<<__FILE__<<", "<<__LINE__<<" hits to be combined have different detIDs, this should not be possible and nothing good will come of it";
    }
    DetId detId = detIdPos.rawId()!=0 ? detIdPos : detIdNeg;
    matchInfo_.push_back(MatchInfo(detId,dRZPos,dRZNeg,dPhiPos,dPhiNeg));
  }
}

TrajSeedMatcher::MatchingCutsV1::MatchingCutsV1(const edm::ParameterSet& pset):
  dPhiMax_(pset.getParameter<double>("dPhiMax")),
  dRZMax_(pset.getParameter<double>("dRZMax")),
  dRZMaxLowEtThres_(pset.getParameter<double>("dRZMaxLowEtThres")),
  dRZMaxLowEtEtaBins_(pset.getParameter<std::vector<double> >("dRZMaxLowEtEtaBins")),
  dRZMaxLowEt_(pset.getParameter<std::vector<double> >("dRZMaxLowEt"))
{
  if(dRZMaxLowEtEtaBins_.size()+1!=dRZMaxLowEt_.size()){
    throw cms::Exception("InvalidConfig")<<" dRZMaxLowEtEtaBins should be 1 less than dRZMaxLowEt when its "<<dRZMaxLowEtEtaBins_.size()<<" vs "<<dRZMaxLowEt_.size();
  }
}

bool TrajSeedMatcher::MatchingCutsV1::operator()(const TrajSeedMatcher::SCHitMatch& scHitMatch)const
{
  if(dPhiMax_>=0 && std::abs(scHitMatch.dPhi()) > dPhiMax_) return false;
  
  const float dRZMax = getDRZCutValue(scHitMatch.et(),scHitMatch.eta());
  if(dRZMax_>=0 && std::abs(scHitMatch.dRZ()) > dRZMax) return false;
	       
  return true;
}

float TrajSeedMatcher::MatchingCutsV1::getDRZCutValue(const float scEt, const float scEta)const
{
  if(scEt>=dRZMaxLowEtThres_) return dRZMax_;
  else{
    const float absEta = std::abs(scEta);
    for(size_t etaNr=0;etaNr<dRZMaxLowEtEtaBins_.size();etaNr++){
      if(absEta<dRZMaxLowEtEtaBins_[etaNr]) return dRZMaxLowEt_[etaNr];
    }
    return dRZMaxLowEt_.back();
  }
}

TrajSeedMatcher::MatchingCutsV2::MatchingCutsV2(const edm::ParameterSet& pset):
  dPhiHighEt_(pset.getParameter<std::vector<double> >("dPhiMaxHighEt")),
  dPhiHighEtThres_(pset.getParameter<std::vector<double> >("dPhiMaxHighEtThres")),
  dPhiLowEtGrad_(pset.getParameter<std::vector<double> >("dPhiMaxLowEtGrad")),
  dRZHighEt_(pset.getParameter<std::vector<double> >("dRZMaxHighEt")),
  dRZHighEtThres_(pset.getParameter<std::vector<double> >("dRZMaxHighEtThres")),
  dRZLowEtGrad_(pset.getParameter<std::vector<double> >("dRZMaxLowEtGrad")),
  etaBins_(pset.getParameter<std::vector<double> >("etaBins"))
{
  auto binSizeCheck = [](size_t sizeEtaBins,const std::vector<double>& vec,const std::string& name){
    if(vec.size()!=sizeEtaBins+1){ 
      throw cms::Exception("InvalidConfig")<<" when constructing TrajSeedMatcher::MatchingCutsV2 "<< name<<" has "<<vec.size()<<" bins, it should be equal to #bins of etaBins+1"<<sizeEtaBins+1;
    }
  };
  binSizeCheck(etaBins_.size(),dPhiHighEt_,"dPhiMaxHighEt");
  binSizeCheck(etaBins_.size(),dPhiHighEtThres_,"dPhiMaxHighEtThres");
  binSizeCheck(etaBins_.size(),dPhiLowEtGrad_,"dPhiMaxLowEtGrad");
  binSizeCheck(etaBins_.size(),dRZHighEt_,"dRZMaxHighEt");
  binSizeCheck(etaBins_.size(),dRZHighEtThres_,"dRZMaxHighEtThres");
  binSizeCheck(etaBins_.size(),dRZLowEtGrad_,"dRZMaxLowEtGrad");
}

bool TrajSeedMatcher::MatchingCutsV2::operator()(const TrajSeedMatcher::SCHitMatch& scHitMatch)const
{
  size_t binNr=getBinNr(scHitMatch.eta());
  float dPhiMax = getCutValue(scHitMatch.et(),dPhiHighEt_[binNr],dPhiHighEtThres_[binNr],dPhiLowEtGrad_[binNr]); 
  if(dPhiMax>=0 && std::abs(scHitMatch.dPhi()) > dPhiMax) return false;  
  float dRZMax = getCutValue(scHitMatch.et(),dRZHighEt_[binNr],dRZHighEtThres_[binNr],dRZLowEtGrad_[binNr]);
  if(dRZMax>=0 && std::abs(scHitMatch.dRZ()) > dRZMax) return false;
  
  return true;
}

//eta bins is exactly 1 smaller than the vectors which will be accessed by this bin nr
size_t TrajSeedMatcher::MatchingCutsV2::getBinNr(float eta)const
{
  const float absEta = std::abs(eta);
  for(size_t etaNr=0;etaNr<etaBins_.size();etaNr++){
    if(absEta<etaBins_[etaNr]) return etaNr;
  }
  return etaBins_.size();
}

