#include "L1Trigger/VertexFinder/interface/AnalysisSettings.h"

namespace l1tVertexFinder {

  ///=== Get configuration parameters

  AnalysisSettings::AnalysisSettings(const edm::ParameterSet& iConfig)
      : AlgoSettings(iConfig),

        // Parameter sets for differents types of configuration parameter.
        genCuts_(iConfig.getParameter<edm::ParameterSet>("GenCuts")),
        l1TrackDef_(iConfig.getParameter<edm::ParameterSet>("L1TrackDef")),
        trackMatchDef_(iConfig.getParameter<edm::ParameterSet>("TrackMatchDef")),

        //=== Cuts on MC truth tracks used for tracking efficiency measurements.
        genMinPt_(genCuts_.getParameter<double>("GenMinPt")),
        genMaxAbsEta_(genCuts_.getParameter<double>("GenMaxAbsEta")),
        genMaxVertR_(genCuts_.getParameter<double>("GenMaxVertR")),
        genMaxVertZ_(genCuts_.getParameter<double>("GenMaxVertZ")),
        genMinStubLayers_(genCuts_.getParameter<unsigned int>("GenMinStubLayers")),

        //=== Rules for deciding when the track finding has found an L1 track candidate
        useLayerID_(l1TrackDef_.getParameter<bool>("UseLayerID")),

        //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
        minFracMatchStubsOnReco_(trackMatchDef_.getParameter<double>("MinFracMatchStubsOnReco")),
        minFracMatchStubsOnTP_(trackMatchDef_.getParameter<double>("MinFracMatchStubsOnTP")),
        minNumMatchLayers_(trackMatchDef_.getParameter<unsigned int>("MinNumMatchLayers")),
        minNumMatchPSLayers_(trackMatchDef_.getParameter<unsigned int>("MinNumMatchPSLayers")),
        stubMatchStrict_(trackMatchDef_.getParameter<bool>("StubMatchStrict")) {
    // If user didn't specify any PDG codes, use e,mu,pi,K,p, to avoid picking up unstable particles like Xi-.
    std::vector<unsigned int> genPdgIdsUnsigned(genCuts_.getParameter<std::vector<unsigned int>>("GenPdgIds"));
    std::vector<int> genPdgIdsAll_ = {11, 13, 211, 321, 2212};
    if (genPdgIdsUnsigned.empty()) {
      genPdgIds_.insert(genPdgIds_.begin(), genPdgIdsAll_.begin(), genPdgIdsAll_.end());
    } else {
      genPdgIds_.insert(genPdgIds_.begin(), genPdgIdsUnsigned.begin(), genPdgIdsUnsigned.end());
    }

    // For simplicity, user need not distinguish particles from antiparticles in configuration file.
    // But here we must store both explicitely in Settings, since TrackingParticleSelector expects them.
    std::transform(genPdgIds_.begin(), genPdgIds_.end(), std::back_inserter(genPdgIds_), std::negate<int>());
    std::transform(genPdgIdsAll_.begin(), genPdgIdsAll_.end(), std::back_inserter(genPdgIdsAll_), std::negate<int>());

    // Define the settings for the TrackingParticleSelectors
    tpSelectorUseForVtxReco_ = TrackingParticleSelector(genMinPt(),
                                                        99999999.,
                                                        -genMaxAbsEta(),
                                                        genMaxAbsEta(),
                                                        genMaxVertR(),
                                                        genMaxVertZ(),
                                                        0,
                                                        true,  //useOnlyTPfromPhysicsCollision
                                                        true,  //useOnlyInTimeParticles
                                                        true,
                                                        false,
                                                        genPdgIds());
    // Range big enough to include all TP needed to measure tracking efficiency
    // and big enough to include any TP that might be reconstructed for fake rate measurement.
    // Include all possible particle types here, as if some are left out, L1 tracks matching one of missing types will be declared fake.
    tpSelectorUse_ = TrackingParticleSelector(std::min(genMinPt(), 2.),
                                              9999999999,
                                              -std::max(genMaxAbsEta(), 3.5),
                                              std::max(genMaxAbsEta(), 3.5),
                                              std::max(10.0, genMaxVertR()),
                                              std::max(35.0, genMaxVertZ()),
                                              0,
                                              false,  //useOnlyTPfromPhysicsCollisionFalse
                                              false,  //useOnlyInTimeParticles
                                              true,
                                              false,
                                              genPdgIds(true));
    tpSelectorUseForEff_ = TrackingParticleSelector(genMinPt(),
                                                    9999999999,
                                                    -genMaxAbsEta(),
                                                    genMaxAbsEta(),
                                                    genMaxVertR(),
                                                    genMaxVertZ(),
                                                    0,
                                                    true,  //useOnlyTPfromPhysicsCollision
                                                    true,  //useOnlyInTimeParticles
                                                    true,
                                                    false,
                                                    genPdgIds());

    //--- Sanity checks
    if (minNumMatchLayers_ > genMinStubLayers_)
      throw cms::Exception(
          "Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type C.");
  }

}  // end namespace l1tVertexFinder
