#ifndef RecoTauTag_RecoTau_DeepTauScaling_h
#define RecoTauTag_RecoTau_DeepTauScaling_h

namespace deep_tau {
  namespace Scaling {
    constexpr float inf = std::numeric_limits<float>::infinity();
    enum class FeatureT{TauFlat, GridGlobal, PfCand_electron, PfCand_muon, PfCand_chHad, PfCand_nHad, PfCand_gamma, Electron, Muon};
    struct ScalingParams
    {
        const std::vector<std::vector<float>> mean_;
        const std::vector<std::vector<float>> std_;
        const std::vector<std::vector<float>> lim_min_;
        const std::vector<std::vector<float>> lim_max_;
    };
  };
};

#endif