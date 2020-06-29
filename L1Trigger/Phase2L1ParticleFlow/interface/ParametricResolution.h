#ifndef L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h
#define L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <cmath>

namespace l1tpf {

  class ParametricResolution {
  public:
    static std::vector<float> getVFloat(const edm::ParameterSet &cpset, const std::string &name);

    ParametricResolution() {}
    ParametricResolution(const edm::ParameterSet &cpset);

    float operator()(const float pt, const float abseta) const;

  protected:
    std::vector<float> etas_, offsets_, scales_, ptMins_, ptMaxs_;
    enum class Kind { Calo, Track };
    Kind kind_;
  };

};  // namespace l1tpf

#endif
