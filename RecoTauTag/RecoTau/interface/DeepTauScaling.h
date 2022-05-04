#ifndef RecoTauTag_RecoTau_DeepTauScaling_h
#define RecoTauTag_RecoTau_DeepTauScaling_h

namespace deep_tau {
  namespace Scaling {
    constexpr float inf = std::numeric_limits<float>::infinity();
    enum class FeatureT{TauFlat, GridGlobal, PfCand_electron, PfCand_muon, PfCand_chHad, PfCand_nHad, PfCand_gamma, Electron, Muon};

    struct ScalingParams
    {
        std::vector<float> mean_;
        std::vector<float> std_;
        std::vector<float> lim_min_;
        std::vector<float> lim_max_;

        template <typename T>
        static float getValue(T value) {
        return std::isnormal(value) ? static_cast<float>(value) : 0.f;
        }

        template<typename T>
        float scale(T value, int var_index) const{
            const float fixed_value = getValue(value);
            const float mean = mean_.at(var_index);
            const float std = std_.at(var_index);
            const float lim_min = lim_min_.at(var_index);
            const float lim_max = lim_max_.at(var_index);
            const float norm_value = (fixed_value - mean) / std;
            return std::clamp(norm_value, lim_min, lim_max);        
        };
    };
  }; // Scaling namespace
}; // deep_tau namespace

#endif