#ifndef __L1Trigger_VertexFinder_AnalysisSettings_h__
#define __L1Trigger_VertexFinder_AnalysisSettings_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/VertexFinder/interface/AlgoSettings.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include <algorithm>
#include <functional>
#include <vector>

namespace l1tVertexFinder {

  class AnalysisSettings : public AlgoSettings {
  public:
    AnalysisSettings(const edm::ParameterSet& iConfig);
    ~AnalysisSettings() {}

    //=== Cuts on MC truth tracks for tracking efficiency measurements.

    double genMinPt() const { return genMinPt_; }
    double genMaxAbsEta() const { return genMaxAbsEta_; }
    double genMaxVertR() const { return genMaxVertR_; }
    double genMaxVertZ() const { return genMaxVertZ_; }
    const std::vector<int>& genPdgIds(bool all = false) const { return (all ? genPdgIdsAll_ : genPdgIds_); }
    // Additional cut on MC truth tracks for algorithmic tracking efficiency measurements.
    unsigned int genMinStubLayers() const { return genMinStubLayers_; }  // Min. number of layers TP made stub in.

    //=== Selection of MC truth tracks.
    const TrackingParticleSelector& tpsUseForVtxReco() const { return tpSelectorUseForVtxReco_; }
    const TrackingParticleSelector& tpsUse() const { return tpSelectorUse_; }
    const TrackingParticleSelector& tpsUseForEff() const { return tpSelectorUseForEff_; }

    //=== Rules for deciding when the track finding has found an L1 track candidate

    // Define layers using layer ID (true) or by bins in radius of 5 cm width (false)?
    bool useLayerID() const { return useLayerID_; }

    //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).

    //--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, set two of them to ultra loose).
    // Min. fraction of matched stubs relative to number of stubs on reco track.
    double minFracMatchStubsOnReco() const { return minFracMatchStubsOnReco_; }
    // Min. fraction of matched stubs relative to number of stubs on tracking particle.
    double minFracMatchStubsOnTP() const { return minFracMatchStubsOnTP_; }
    // Min. number of matched layers & min. number of matched PS layers..
    unsigned int minNumMatchLayers() const { return minNumMatchLayers_; }
    unsigned int minNumMatchPSLayers() const { return minNumMatchPSLayers_; }
    // Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
    bool stubMatchStrict() const { return stubMatchStrict_; }

  private:
    // Parameter sets for differents types of configuration parameter.
    edm::ParameterSet genCuts_;
    edm::ParameterSet l1TrackDef_;
    edm::ParameterSet trackMatchDef_;

    // Cuts on truth tracking particles.
    double genMinPt_;
    double genMaxAbsEta_;
    double genMaxVertR_;
    double genMaxVertZ_;
    std::vector<int> genPdgIds_;
    std::vector<int> genPdgIdsAll_;
    unsigned int genMinStubLayers_;

    // Rules for deciding when the track-finding has found an L1 track candidate
    bool useLayerID_;

    // Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
    double minFracMatchStubsOnReco_;
    double minFracMatchStubsOnTP_;
    unsigned int minNumMatchLayers_;
    unsigned int minNumMatchPSLayers_;
    bool stubMatchStrict_;

    // Track Fitting Settings
    std::vector<std::string> trackFitters_;
    double chi2OverNdfCut_;
    bool detailedFitOutput_;
    unsigned int numTrackFitIterations_;
    bool killTrackFitWorstHit_;
    double generalResidualCut_;
    double killingResidualCut_;

    // Tracking particle selectors
    TrackingParticleSelector tpSelectorUseForVtxReco_;
    TrackingParticleSelector tpSelectorUse_;
    TrackingParticleSelector tpSelectorUseForEff_;
  };

}  // end namespace l1tVertexFinder

#endif
