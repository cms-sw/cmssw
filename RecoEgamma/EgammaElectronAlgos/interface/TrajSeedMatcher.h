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
#include "RecoEgamma/EgammaElectronAlgos/interface/utils.h"

#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"

namespace edm {
  class ConsumesCollector;
  class EventSetup;
  class ConfigurationDescriptions;
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

class FreeTrajectoryState;
class TrackingRecHit;

class TrajSeedMatcher {
public:
  struct SCHitMatch {
    const DetId detId = 0;
    const GlobalPoint hitPos;
    const float dRZ = std::numeric_limits<float>::max();
    const float dPhi = std::numeric_limits<float>::max();
    const TrackingRecHit& hit;
    const float et = 0.f;
    const float eta = 0.f;
    const float phi = 0.f;
    const int charge = 0;
    const int nrClus = 0;
  };

  struct MatchInfo {
  public:
    DetId detId;
    float dRZPos, dRZNeg;
    float dPhiPos, dPhiNeg;

    MatchInfo(const DetId& iDetId, float iDRZPos, float iDRZNeg, float iDPhiPos, float iDPhiNeg)
        : detId(iDetId), dRZPos(iDRZPos), dRZNeg(iDRZNeg), dPhiPos(iDPhiPos), dPhiNeg(iDPhiNeg) {}
  };

  class SeedWithInfo {
  public:
    SeedWithInfo(const TrajectorySeed& seed,
                 const std::vector<SCHitMatch>& posCharge,
                 const std::vector<SCHitMatch>& negCharge,
                 int nrValidLayers);
    ~SeedWithInfo() = default;

    const TrajectorySeed& seed() const { return seed_; }
    float dRZPos(size_t hitNr) const { return getVal(hitNr, &MatchInfo::dRZPos); }
    float dRZNeg(size_t hitNr) const { return getVal(hitNr, &MatchInfo::dRZNeg); }
    float dPhiPos(size_t hitNr) const { return getVal(hitNr, &MatchInfo::dPhiPos); }
    float dPhiNeg(size_t hitNr) const { return getVal(hitNr, &MatchInfo::dPhiNeg); }
    DetId detId(size_t hitNr) const { return hitNr < matchInfo_.size() ? matchInfo_[hitNr].detId : DetId(0); }
    size_t nrMatchedHits() const { return matchInfo_.size(); }
    const std::vector<MatchInfo>& matches() const { return matchInfo_; }
    int nrValidLayers() const { return nrValidLayers_; }

  private:
    float getVal(size_t hitNr, float MatchInfo::*val) const {
      return hitNr < matchInfo_.size() ? matchInfo_[hitNr].*val : std::numeric_limits<float>::max();
    }

  private:
    const TrajectorySeed& seed_;
    std::vector<MatchInfo> matchInfo_;
    int nrValidLayers_;
  };

  class MatchingCuts {
  public:
    MatchingCuts() {}
    virtual ~MatchingCuts() {}
    virtual bool operator()(const SCHitMatch& scHitMatch) const = 0;
  };

  class MatchingCutsV1 : public MatchingCuts {
  public:
    explicit MatchingCutsV1(const edm::ParameterSet& pset);
    bool operator()(const SCHitMatch& scHitMatch) const override;

  private:
    float getDRZCutValue(const float scEt, const float scEta) const;

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
    bool operator()(const SCHitMatch& scHitMatch) const override;

  private:
    size_t getBinNr(float eta) const;
    float getCutValue(float et, float highEt, float highEtThres, float lowEtGrad) const {
      return highEt + std::min(0.f, et - highEtThres) * lowEtGrad;
    }

  private:
    std::vector<double> dPhiHighEt_, dPhiHighEtThres_, dPhiLowEtGrad_;
    std::vector<double> dRZHighEt_, dRZHighEtThres_, dRZLowEtGrad_;
    std::vector<double> etaBins_;
  };

public:
  struct Configuration {
    Configuration(const edm::ParameterSet& pset, edm::ConsumesCollector&& cc);

    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> paramMagFieldToken;
    const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navSchoolToken;
    const edm::ESGetToken<DetLayerGeometry, RecoGeometryRecord> detLayerGeomToken;

    const bool useRecoVertex;
    const bool enableHitSkipping;
    const bool requireExactMatchCount;
    const bool useParamMagFieldIfDefined;

    //these two variables determine how hits we require
    //based on how many valid layers we had
    //right now we always need atleast two hits
    //also highly dependent on the seeds you pass in
    //which also require a given number of hits
    const std::vector<unsigned int> minNrHits;
    const std::vector<int> minNrHitsValidLayerBins;

    const std::vector<std::unique_ptr<MatchingCuts> > matchingCuts;
  };

  explicit TrajSeedMatcher(Configuration const& cfg,
                           edm::EventSetup const& iSetup,
                           MeasurementTrackerEvent const& measTkEvt);
  ~TrajSeedMatcher() = default;

  static edm::ParameterSetDescription makePSetDescription();

  std::vector<TrajSeedMatcher::SeedWithInfo> operator()(const TrajectorySeedCollection& seeds,
                                                        const GlobalPoint& candPos,
                                                        const GlobalPoint& vprim,
                                                        const float energy);

private:
  std::vector<SCHitMatch> processSeed(const TrajectorySeed& seed,
                                      const GlobalPoint& candPos,
                                      const GlobalPoint& vprim,
                                      const float energy,
                                      const TrajectoryStateOnSurface& initialTrajState);

  static float getZVtxFromExtrapolation(const GlobalPoint& primeVtxPos,
                                        const GlobalPoint& hitPos,
                                        const GlobalPoint& candPos);

  const TrajectoryStateOnSurface& getTrajStateFromVtx(const TrackingRecHit& hit,
                                                      const TrajectoryStateOnSurface& initialState,
                                                      const PropagatorWithMaterial& propagator);

  const TrajectoryStateOnSurface& getTrajStateFromPoint(const TrackingRecHit& hit,
                                                        const FreeTrajectoryState& initialState,
                                                        const GlobalPoint& point,
                                                        const PropagatorWithMaterial& propagator);

  TrajectoryStateOnSurface makeTrajStateOnSurface(const GlobalPoint& pos,
                                                  const GlobalPoint& vtx,
                                                  const float energy,
                                                  const int charge) const;
  void clearCache();

  bool passesMatchSel(const SCHitMatch& hit, const size_t hitNr) const;

  int getNrValidLayersAlongTraj(const SCHitMatch& hit1,
                                const SCHitMatch& hit2,
                                const GlobalPoint& candPos,
                                const GlobalPoint& vprim,
                                const float energy,
                                const int charge);

  int getNrValidLayersAlongTraj(const DetId& hitId, const TrajectoryStateOnSurface& hitTrajState) const;

  bool layerHasValidHits(const DetLayer& layer,
                         const TrajectoryStateOnSurface& hitSurState,
                         const Propagator& propToLayerFromState) const;

  size_t getNrHitsRequired(const int nrValidLayers) const;

  //parameterised b-fields may not be valid for entire detector, just tracker volume
  //however need we ecal so we auto select based on the position
  const MagneticField& getMagField(const GlobalPoint& point) const {
    return cfg_.useParamMagFieldIfDefined && magFieldParam_.isDefined(point) ? magFieldParam_ : magField_;
  }

private:
  static constexpr float kElectronMass_ = 0.000511;

  Configuration const& cfg_;

  MagneticField const& magField_;
  MagneticField const& magFieldParam_;
  MeasurementTrackerEvent const& measTkEvt_;
  NavigationSchool const& navSchool_;
  DetLayerGeometry const& detLayerGeom_;

  PropagatorWithMaterial forwardPropagator_;
  PropagatorWithMaterial backwardPropagator_;

  std::unordered_map<int, TrajectoryStateOnSurface> trajStateFromVtxPosChargeCache_;
  std::unordered_map<int, TrajectoryStateOnSurface> trajStateFromVtxNegChargeCache_;

  IntGlobalPointPairUnorderedMap<TrajectoryStateOnSurface> trajStateFromPointPosChargeCache_;
  IntGlobalPointPairUnorderedMap<TrajectoryStateOnSurface> trajStateFromPointNegChargeCache_;
};

#endif
