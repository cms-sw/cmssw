#ifndef L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h
#define L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h

#include <string>
#include <vector>
#include <cmath>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1tpf {

  class ParametricResolution {
  public:
    enum class Kind { Calo, Track };

    ParametricResolution() {}
    ParametricResolution(const edm::ParameterSet &cpset);
    ParametricResolution(Kind kind,
                         std::vector<float> etas,
                         std::vector<float> offsets,
                         std::vector<float> scales,
                         std::vector<float> ptMins,
                         std::vector<float> ptMaxs)
        : kind_(kind), etas_(etas), offsets_(offsets), scales_(scales), ptMins_(ptMins), ptMaxs_(ptMaxs) {};

    float operator()(const float pt, const float abseta) const;

  protected:
    static std::vector<float> getVFloat(const edm::ParameterSet &cpset, const std::string &name);

    Kind kind_;
    std::vector<float> etas_, offsets_, scales_, ptMins_, ptMaxs_;
  };

};  // namespace l1tpf

#endif
