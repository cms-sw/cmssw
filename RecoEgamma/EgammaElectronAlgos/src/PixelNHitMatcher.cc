#include "RecoEgamma/EgammaElectronAlgos/interface/PixelNHitMatcher.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h" //might remove

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

class DetIdTools
{
public:
  static const int kDetOffset = 28;
  static const int kSubDetOffset = 25;

  static const int kDetMask = 0xF << kDetOffset;
  static const int kSubDetMask  = 0x7 << kSubDetOffset;
  
  static const int kBPXLayerOffset     = 20;
  static const int kBPXLadderOffset    = 12;
  static const int kBPXModuleOffset    = 2;
  static const int kBPXLayerMask       = 0xF;
  static const int kBPXLadderMask      = 0xFF;
  static const int kBPXModuleMask      = 0xFF;
  
  static const int kFPXSideOffset      = 23;
  static const int kFPXDiskOffset      = 18;
  static const int kFPXBladeOffset     = 12;   
  static const int kFPXPanelOffset     = 10; 
  static const int kFPXModuleOffset    = 2;   
  static const int kFPXSideMask        = 0x3;
  static const int kFPXDiskMask        = 0xF;
  static const int kFPXBladeMask       = 0x3F;   
  static const int kFPXPanelMask       = 0x3; 
  static const int kFPXModuleMask      = 0xFF; 

   //pixel tools
  static int getVal(int detId,int offset,int mask){return (detId>>offset)&mask;}
  static int layerBPX(int detId){return getVal(detId,kBPXLayerOffset,kBPXLayerMask);}
  static int ladderBPX(int detId){return getVal(detId,kBPXLadderOffset,kBPXLadderMask);}
  static int moduleBPX(int detId){return getVal(detId,kBPXModuleOffset,kBPXModuleMask);}
  static int sideFPX(int detId){return getVal(detId,kFPXSideOffset,kFPXSideMask);} 
  static int diskFPX(int detId){return getVal(detId,kFPXDiskOffset,kFPXDiskMask);} 
 
};
namespace{
  int getLayerOrDisk(DetId id){
    return id.subdetId()==1 ? DetIdTools::layerBPX(id.rawId()) : DetIdTools::diskFPX(id.rawId());
  }
}

PixelNHitMatcher::PixelNHitMatcher(const edm::ParameterSet& pset):
  cacheIDMagField_(0)
{
  useRecoVertex_ = pset.getParameter<bool>("useRecoVertex");
  navSchoolLabel_ = "SimpleNavigationSchool";
  detLayerGeomLabel_ = "hltESPGlobalDetLayerGeometry";
  const auto cutsPSets=pset.getParameter<std::vector<edm::ParameterSet> >("matchingCuts");
  for(const auto & cutPSet : cutsPSets){
    matchingCuts_.push_back(MatchingCuts(cutPSet));
  }
  nrHitsRequired_=matchingCuts_.size();
}

void PixelNHitMatcher::fillDescriptions(edm::ConfigurationDescriptions& description)
{
  edm::ParameterSetDescription desc;
  desc.add<bool>("useRecoVertex",false);

  edm::ParameterSetDescription cutsDesc;
  cutsDesc.add<double>("dPhiMax",0.04);
  cutsDesc.add<double>("dZMax",0.04);
  cutsDesc.add<double>("dRIMax",0.04);
  cutsDesc.add<double>("dRFMax",0.04);
  desc.addVPSet("matchingCuts",cutsDesc);
  description.add("pixelNHitMatch",desc);
}

void PixelNHitMatcher::doEventSetup(const edm::EventSetup& iSetup)
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


std::vector<PixelNHitMatcher::SeedWithInfo>
PixelNHitMatcher::compatibleSeeds(const TrajectorySeedCollection& seeds, const GlobalPoint& candPos,
				  const GlobalPoint & vprim, const float energy)
{
  if(!forwardPropagator_ || !backwardPropagator_ || !magField_.isValid()){
    throw cms::Exception("LogicError") <<__FUNCTION__<<" can not make pixel seeds as event setup has not properly been called";
  }

  clearCache();
  
  std::vector<SeedWithInfo> matchedSeeds;
  for(const auto& seed : seeds) {
    std::vector<HitInfo> matchedHitsNeg = processSeed(seed,candPos,vprim,energy,-1);
    std::vector<HitInfo> matchedHitsPos = processSeed(seed,candPos,vprim,energy,+1);
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

    size_t nrHitsRequired = nrValidLayers>=4 ? 3 : 2;
    //   std::cout <<"nr valid layers "<<nrValidLayers<<" nr hits "<<matchedHitsNeg.size()<<" pos "<<matchedHitsPos.size();//<<std::endl;

    if(matchedHitsNeg.size()==nrHitsRequired ||
       matchedHitsPos.size()==nrHitsRequired){
      //do the result
      //    std::cout <<"acceped "<<std::endl;
      matchedSeeds.push_back({seed,matchedHitsPos,matchedHitsNeg,nrValidLayers});
    }//else std::cout <<"rejected "<<std::endl;
    

  }
  return matchedSeeds;
}

//checks if the hits of the seed match within requested selection
//matched hits are required to be consecutive, as soon as hit isnt matched,
//the function returns, it doesnt allow skipping hits
std::vector<PixelNHitMatcher::HitInfo>
PixelNHitMatcher::processSeed(const TrajectorySeed& seed, const GlobalPoint& candPos,
			      const GlobalPoint & vprim, const float energy, const int charge )
{
  
  // if(seed.nHits()!=nrHitsRequired_){
  //   throw cms::Exception("Configuration") <<"PixelNHitMatcher is being fed seeds with "<<seed.nHits()<<" but requires "<<nrHitsRequired_<<" for a match, it is inconsistantly configured";
  // }

  
  FreeTrajectoryState trajStateFromVtx = FTSFromVertexToPointFactory::get(*magField_, candPos, vprim, energy, charge);
  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface initialTrajState(trajStateFromVtx,*bpb(trajStateFromVtx.position(), 
								  trajStateFromVtx.momentum()));
 
  std::vector<HitInfo> matchedHits;
  HitInfo firstHit = matchFirstHit(seed,initialTrajState,vprim,*backwardPropagator_);
  if(passesMatchSel(firstHit,0)){
    matchedHits.push_back(firstHit);

    //now we can figure out the z vertex
    double zVertex = useRecoVertex_ ? vprim.z() : getZVtxFromExtrapolation(vprim,firstHit.pos(),candPos);
    GlobalPoint vertex(vprim.x(),vprim.y(),zVertex);
    
    //FIXME: rename this variable
    FreeTrajectoryState fts2 = FTSFromVertexToPointFactory::get(*magField_, firstHit.pos(), 
								vertex, energy, charge) ;
 
    GlobalPoint prevHitPos = firstHit.pos();
    for(size_t hitNr=1;hitNr<nrHitsRequired_ && hitNr<seed.nHits();hitNr++){
      HitInfo hit = match2ndToNthHit(seed,fts2,hitNr,prevHitPos,vertex,*forwardPropagator_);
      if(passesMatchSel(hit,hitNr)){
	matchedHits.push_back(hit);
	prevHitPos = hit.pos();
      }else break;
    }
  }
  return matchedHits;
}

// compute the z vertex from the candidate position and the found pixel hit
float PixelNHitMatcher::getZVtxFromExtrapolation(const GlobalPoint& primeVtxPos,const GlobalPoint& hitPos,
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

bool PixelNHitMatcher::passTrajPreSel(const GlobalPoint& hitPos,const GlobalPoint& candPos)const
{
  float dt = hitPos.x()*candPos.x()+hitPos.y()*candPos.y();
  if (dt<0) return false;
  if (dt<kPhiCut_*(candPos.perp()*hitPos.perp())) return false;
  return true;
}

const TrajectoryStateOnSurface& PixelNHitMatcher::getTrajStateFromVtx(const TrackingRecHit& hit,const TrajectoryStateOnSurface& initialState,const PropagatorWithMaterial& propagator)
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

const TrajectoryStateOnSurface& PixelNHitMatcher::getTrajStateFromPoint(const TrackingRecHit& hit,const FreeTrajectoryState& initialState,const GlobalPoint& point,const PropagatorWithMaterial& propagator)
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

PixelNHitMatcher::HitInfo PixelNHitMatcher::matchFirstHit(const TrajectorySeed& seed,const TrajectoryStateOnSurface& initialState,const GlobalPoint& vtxPos,const PropagatorWithMaterial& propagator)
{
  const TrajectorySeed::range& hits = seed.recHits();
  auto hitIt = hits.first;

  if(hitIt->isValid()){
    const TrajectoryStateOnSurface& trajStateFromVtx = getTrajStateFromVtx(*hitIt,initialState,propagator);
    if(trajStateFromVtx.isValid()) return HitInfo(vtxPos,trajStateFromVtx,*hitIt);  
  }
  return HitInfo();
}

PixelNHitMatcher::HitInfo PixelNHitMatcher::match2ndToNthHit(const TrajectorySeed& seed,
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
      return HitInfo(vtxPos,trajState,*hitIt);  
    }
  }
  return HitInfo();
  
}

void PixelNHitMatcher::clearCache()
{
  trajStateFromVtxPosChargeCache_.clear();
  trajStateFromVtxNegChargeCache_.clear();
  trajStateFromPointPosChargeCache_.clear();
  trajStateFromPointNegChargeCache_.clear();
}

bool PixelNHitMatcher::passesMatchSel(const PixelNHitMatcher::HitInfo& hit,const size_t hitNr)const
{
  if(hitNr<matchingCuts_.size()){
    return matchingCuts_[hitNr](hit);
  }else{
    throw cms::Exception("LogicError") <<" Error, attempting to apply selection to hit "<<hitNr<<" but only cuts for "<<matchingCuts_.size()<<" defined";
  }
  
}

int PixelNHitMatcher::getNrValidLayersAlongTraj(const HitInfo& hit1,const HitInfo& hit2,
						const GlobalPoint& candPos,
						const GlobalPoint & vprim, 
						const float energy, const int charge)
{
  double zVertex = useRecoVertex_ ? vprim.z() : getZVtxFromExtrapolation(vprim,hit1.pos(),candPos);
  GlobalPoint vertex(vprim.x(),vprim.y(),zVertex);
  
  //FIXME: rename this variable
  FreeTrajectoryState fts2 = FTSFromVertexToPointFactory::get(*magField_,hit1.pos(), 
							      vertex, energy, charge) ;
  const TrajectoryStateOnSurface& trajState = getTrajStateFromPoint(*hit2.hit(),fts2,hit1.pos(),*forwardPropagator_);
  return getNrValidLayersAlongTraj(hit2.hit()->geographicalId(),trajState); 
}

int PixelNHitMatcher::getNrValidLayersAlongTraj(const DetId& hitId,const TrajectoryStateOnSurface& hitTrajState)const
{
  
  const DetLayer* detLayer = detLayerGeom_->idToLayer(hitId);
  if(detLayer==nullptr) return 0;

  const FreeTrajectoryState& hitFreeState = *hitTrajState.freeState();
  const std::vector<const DetLayer*> inLayers  = navSchool_->compatibleLayers(*detLayer,hitFreeState,oppositeToMomentum); 
  const std::vector<const DetLayer*> outLayers = navSchool_->compatibleLayers(*detLayer,hitFreeState,alongMomentum); 
  
  int nrValidLayers=1; //because our current hit is valid
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
  //  int layerOrDisk=getLayerOrDisk(hitId);
  //  std::cout <<"sub "<<hitId.subdetId()<<" for layer "<<layerOrDisk<<" nr valid "<<nrValidLayers<<" nr pix in "<<nrPixInLayers<<" nr pix out "<<nrPixOutLayers<<std::endl;
  return nrValidLayers;
}
						 
bool PixelNHitMatcher::layerHasValidHits(const DetLayer& layer,const TrajectoryStateOnSurface& hitSurState,
					 const Propagator& propToLayerFromState)const
					 //		    const Estimator& estimator,
//					 const MeasurementTrackerEvent& measTkEvt)
{
  //FIXME: do not hardcode with werid magic numbers stolen from ancient tracking code
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




PixelNHitMatcher::HitInfo::HitInfo(const GlobalPoint& vtxPos,
				   const TrajectoryStateOnSurface& trajState,
				   const TrackingRecHit& hit):
  detId_(hit.geographicalId()),
  pos_(hit.globalPosition()),
  hit_(&hit)
{
  EleRelPointPair pointPair(pos_,trajState.globalParameters().position(),vtxPos);
  dRZ_ = detId_.subdetId()==PixelSubdetector::PixelBarrel ? pointPair.dZ() : pointPair.dPerp();
  dPhi_ = pointPair.dPhi();
}
    

PixelNHitMatcher::SeedWithInfo::
SeedWithInfo(const TrajectorySeed& seed,
	     const std::vector<HitInfo>& posCharge,
	     const std::vector<HitInfo>& negCharge,
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
      cms::Exception("LogicError")<<" error in "<<__FILE__<<", "<<__LINE__<<" hits to be combined have different detIDs, this should not be possible and will cause Bad Things (tm) to happen";
    }
    DetId detId = detIdPos.rawId()!=0 ? detIdPos : detIdNeg;
    matchInfo_.push_back(MatchInfo(detId,dRZPos,dRZNeg,dPhiPos,dPhiNeg));
  }
}

PixelNHitMatcher::MatchingCuts::MatchingCuts(const edm::ParameterSet& pset):
  dPhiMax_(pset.getParameter<double>("dPhiMax")),
  dZMax_(pset.getParameter<double>("dZMax")),
  dRIMax_(pset.getParameter<double>("dRIMax")),
  dRFMax_(pset.getParameter<double>("dRFMax"))
{
  
}

bool PixelNHitMatcher::MatchingCuts::operator()(const PixelNHitMatcher::HitInfo& hit)const
{
  if(std::abs(hit.dPhi()) > dPhiMax_) return false;
  float dZOrRMax = hit.subdetId()==PixelSubdetector::PixelBarrel ? dZMax_ : dRFMax_;
  if(std::abs(hit.dRZ()) > dZOrRMax) return false;
  
  return true;
}
