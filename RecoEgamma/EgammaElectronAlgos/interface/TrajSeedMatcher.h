#ifndef RecoEgamma_EgammaElectronAlgos_TrajSeedMatcher_h
#define RecoEgamma_EgammaElectronAlgos_TrajSeedMatcher_h


//******************************************************************************
//
// Part of the refactorisation of of the E/gamma pixel matching for 2017 pixels
// This refactorisation converts the monolithic  approach to a series of 
// independent producer modules, with each modules performing  a specific 
// job as recommended by the 2017 tracker framework
//
//
// The module is based of PixelHitMatcher (the seed based functions) but
// extended to match on an arbitary number of hits rather than just doublets.
// It is also aware of how many layers the supercluster trajectory passed through
// and uses that information to determine how many hits to require
// Other than that, its a direct port and follows what PixelHitMatcher did
// 
//
// Author : Sam Harper (RAL), 2017
//
//*******************************************************************************




#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/RecoGeometry/interface/GlobalDetLayerGeometry.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include <unordered_map>

namespace edm{
  class EventSetup;
  class ConfigurationDescriptions;
  class ParameterSet;
  class ParameterSetDescription;
}

class FreeTrajectoryState;
class TrackingRecHit;


//stolen from PixelHitMatcher
//decide if its evil or not later
//actually I think the answer is, yes, yes its evil
//maybe replace with less evil?
namespace std{
  template<>
    struct hash<std::pair<int,GlobalPoint> > {
    std::size_t operator()(const std::pair<int,GlobalPoint>& g) const {
      auto h1 = std::hash<unsigned long long>()((unsigned long long)g.first);
      unsigned long long k; memcpy(&k, &g.second,sizeof(k));
      auto h2 = std::hash<unsigned long long>()(k);
      return h1 ^ (h2 << 1);
      }
  };
  template<>
  struct equal_to<std::pair<int,GlobalPoint> > : public std::binary_function<std::pair<int,GlobalPoint>,std::pair<int,GlobalPoint>,bool> {
    bool operator()(const std::pair<int,GlobalPoint>& a, 
		    const std::pair<int,GlobalPoint>& b)  const {
      return (a.first == b.first) & (a.second == b.second);
    }
  };
}

class TrajSeedMatcher {
public:
  class SCHitMatch {
  public:
    SCHitMatch():detId_(0),
	      dRZ_(std::numeric_limits<float>::max()),
	      dPhi_(std::numeric_limits<float>::max()),
	      hit_(nullptr),
	      et_(0),eta_(0),phi_(0),charge_(0),nrClus_(0){}

    //does not set charge,et,nrclus
    SCHitMatch(const GlobalPoint& vtxPos,
	    const TrajectoryStateOnSurface& trajState,
	    const TrackingRecHit& hit
	    );
    ~SCHitMatch()=default;

    void setExtra(float et, float eta, float phi, int charge, int nrClus){
      et_=et;
      eta_=eta;
      phi_=phi;
      charge_=charge;
      nrClus_=nrClus;
    }
    
    int subdetId()const{return detId_.subdetId();}
    DetId detId()const{return detId_;}
    float dRZ()const{return dRZ_;}
    float dPhi()const{return dPhi_;}
    const GlobalPoint& hitPos()const{return hitPos_;}
    float et()const{return et_;}
    float eta()const{return eta_;}
    float phi()const{return phi_;}
    int charge()const{return charge_;}
    int nrClus()const{return nrClus_;}
    const TrackingRecHit* hit()const{return hit_;}
  private:
    DetId detId_;
    GlobalPoint hitPos_;
    float dRZ_;
    float dPhi_;    
    const TrackingRecHit* hit_; //we do not own this
    //extra quanities which are set later
    float et_;
    float eta_;
    float phi_;
    int charge_;
    int nrClus_;
  };

 

  struct MatchInfo {
  public:
    DetId detId;
    float dRZPos,dRZNeg;
    float dPhiPos,dPhiNeg;
    
    MatchInfo(const DetId& iDetId,
	      float iDRZPos, float iDRZNeg,
	      float iDPhiPos, float iDPhiNeg):
      detId(iDetId),dRZPos(iDRZPos),dRZNeg(iDRZNeg),
      dPhiPos(iDPhiPos),dPhiNeg(iDPhiNeg){}
  };

  class SeedWithInfo {
  public:
    SeedWithInfo(const TrajectorySeed& seed,
		 const std::vector<SCHitMatch>& posCharge,
		 const std::vector<SCHitMatch>& negCharge,
		 int nrValidLayers);
    ~SeedWithInfo()=default;
    
    const TrajectorySeed& seed()const{return seed_;}
    float dRZPos(size_t hitNr)const{return getVal(hitNr,&MatchInfo::dRZPos);}
    float dRZNeg(size_t hitNr)const{return getVal(hitNr,&MatchInfo::dRZNeg);}
    float dPhiPos(size_t hitNr)const{return getVal(hitNr,&MatchInfo::dPhiPos);}
    float dPhiNeg(size_t hitNr)const{return getVal(hitNr,&MatchInfo::dPhiNeg);}
    DetId detId(size_t hitNr)const{return hitNr<matchInfo_.size() ? matchInfo_[hitNr].detId : DetId(0);}
    size_t nrMatchedHits()const{return matchInfo_.size();}
    const std::vector<MatchInfo>& matches()const{return matchInfo_;}
    int nrValidLayers()const{return nrValidLayers_;}
  private:
    float getVal(size_t hitNr,float MatchInfo::*val)const{
      return hitNr<matchInfo_.size() ? matchInfo_[hitNr].*val : std::numeric_limits<float>::max();
    }

  private:
    const TrajectorySeed& seed_;
    std::vector<MatchInfo> matchInfo_;
    int nrValidLayers_;
  };

  class MatchingCuts {
  public:
    MatchingCuts(){}
    virtual ~MatchingCuts(){}
    virtual bool operator()(const SCHitMatch& scHitMatch)const=0;
  };

  class MatchingCutsV1 : public MatchingCuts {
  public:
    explicit MatchingCutsV1(const edm::ParameterSet& pset);
    bool operator()(const SCHitMatch& scHitMatch)const override;
  private:
    float getDRZCutValue(const float scEt, const float scEta)const;
  private:
    const double dPhiMax_;
    const double dRZMax_;
    const double dRZMaxLowEtThres_;
    const std::vector<double> dRZMaxLowEtEtaBins_; 
    const std::vector<double> dRZMaxLowEt_; 
  };

  class MatchingCutsV2 : public MatchingCuts {
  public:
    explicit MatchingCutsV2(const edm::ParameterSet& pset);
    bool operator()(const SCHitMatch& scHitMatch)const override;
  private:
    size_t getBinNr(float eta)const;
    float getCutValue(float et, float highEt, float highEtThres, float lowEtGrad)const{
      return  highEt + std::min(0.f,et-highEtThres)*lowEtGrad;
    }
  private:
    std::vector<double> dPhiHighEt_,dPhiHighEtThres_,dPhiLowEtGrad_;
    std::vector<double> dRZHighEt_,dRZHighEtThres_,dRZLowEtGrad_;
    std::vector<double> etaBins_;
  };


public:  
  explicit TrajSeedMatcher(const edm::ParameterSet& pset);
  ~TrajSeedMatcher()=default;

  static edm::ParameterSetDescription makePSetDescription();

  void doEventSetup(const edm::EventSetup& iSetup);
  
  std::vector<TrajSeedMatcher::SeedWithInfo>
  compatibleSeeds(const TrajectorySeedCollection& seeds, const GlobalPoint& candPos,
		  const GlobalPoint & vprim, const float energy);

  void setMeasTkEvtHandle(edm::Handle<MeasurementTrackerEvent> handle){measTkEvt_=std::move(handle);}
  
private:
  
  std::vector<SCHitMatch> processSeed(const TrajectorySeed& seed, const GlobalPoint& candPos,
				   const GlobalPoint & vprim, const float energy, const int charge );

  static float getZVtxFromExtrapolation(const GlobalPoint& primeVtxPos, const GlobalPoint& hitPos,
					const GlobalPoint& candPos);
  
  bool passTrajPreSel(const GlobalPoint& hitPos,const GlobalPoint& candPos)const;
  
  TrajSeedMatcher::SCHitMatch matchFirstHit(const TrajectorySeed& seed,
					    const TrajectoryStateOnSurface& trajState,
					    const GlobalPoint& vtxPos,
					    const PropagatorWithMaterial& propagator);

  TrajSeedMatcher::SCHitMatch match2ndToNthHit(const TrajectorySeed& seed,
					       const FreeTrajectoryState& trajState,
					       const size_t hitNr,	
					       const GlobalPoint& prevHitPos,
					       const GlobalPoint& vtxPos,
					       const PropagatorWithMaterial& propagator);
  
  const TrajectoryStateOnSurface& getTrajStateFromVtx(const TrackingRecHit& hit,
						      const TrajectoryStateOnSurface& initialState,
						      const PropagatorWithMaterial& propagator);

  const TrajectoryStateOnSurface& getTrajStateFromPoint(const TrackingRecHit& hit,
							const FreeTrajectoryState& initialState,
							const GlobalPoint& point,
							const PropagatorWithMaterial& propagator);

  void clearCache();

  bool passesMatchSel(const SCHitMatch& hit, const size_t hitNr)const;

  int getNrValidLayersAlongTraj(const SCHitMatch& hit1, const SCHitMatch& hit2,
				const GlobalPoint& candPos,
				const GlobalPoint & vprim, 
				const float energy, const int charge);

  int getNrValidLayersAlongTraj(const DetId& hitId,
				const TrajectoryStateOnSurface& hitTrajState)const;
				
  bool layerHasValidHits(const DetLayer& layer,const TrajectoryStateOnSurface& hitSurState,
			 const Propagator& propToLayerFromState)const;
  
  size_t getNrHitsRequired(const int nrValidLayers)const;
    
private:
  static constexpr float kElectronMass_ = 0.000511;
  static constexpr float kPhiCut_ = -0.801144;//cos(2.5)
  std::unique_ptr<PropagatorWithMaterial> forwardPropagator_;
  std::unique_ptr<PropagatorWithMaterial> backwardPropagator_;
  unsigned long long cacheIDMagField_;
  edm::ESHandle<MagneticField> magField_;
  edm::Handle<MeasurementTrackerEvent> measTkEvt_;
  edm::ESHandle<NavigationSchool> navSchool_;
  edm::ESHandle<DetLayerGeometry> detLayerGeom_;
  std::string navSchoolLabel_;
  std::string detLayerGeomLabel_;

  bool useRecoVertex_;
  std::vector<std::unique_ptr<MatchingCuts> > matchingCuts_;
  
  //these two varibles determine how hits we require 
  //based on how many valid layers we had
  //right now we always need atleast two hits
  //also highly dependent on the seeds you pass in 
  //which also require a given number of hits
  const std::vector<unsigned int> minNrHits_;
  const std::vector<int> minNrHitsValidLayerBins_;

  std::unordered_map<int,TrajectoryStateOnSurface> trajStateFromVtxPosChargeCache_;
  std::unordered_map<int,TrajectoryStateOnSurface> trajStateFromVtxNegChargeCache_;

  std::unordered_map<std::pair<int,GlobalPoint>,TrajectoryStateOnSurface> trajStateFromPointPosChargeCache_;
  std::unordered_map<std::pair<int,GlobalPoint>,TrajectoryStateOnSurface> trajStateFromPointNegChargeCache_;

};

#endif
