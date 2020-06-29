#ifndef L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h
#define L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <cmath>

namespace l1tpf {

  class ParametricResolution {
  public:
    static std::vector<float> getVFloat(const edm::ParameterSet &cpset, const std::string &name) {
      std::vector<double> vd = cpset.getParameter<std::vector<double>>(name);
      return std::vector<float>(vd.begin(), vd.end());
    }

    ParametricResolution() {}
    ParametricResolution(const edm::ParameterSet &cpset)
        : etas(getVFloat(cpset, "etaBins")), offsets(getVFloat(cpset, "offset")), scales(getVFloat(cpset, "scale")) {
      if (cpset.existsAs<std::vector<double>>("ptMin")) {
        ptMins = getVFloat(cpset, "ptMin");
      } else {
        float ptMin = cpset.existsAs<double>("ptMin") ? cpset.getParameter<double>("ptMin") : 0;
        ptMins = std::vector<float>(etas.size(), ptMin);
      }
      if (cpset.existsAs<std::vector<double>>("ptMax")) {
        ptMaxs = getVFloat(cpset, "ptMax");
      } else {
        ptMaxs = std::vector<float>(etas.size(), 1e6);
      }
      std::string skind = cpset.getParameter<std::string>("kind");
      if (skind == "track")
        kind = Kind::Track;
      else if (skind == "calo")
        kind = Kind::Calo;
      else
        throw cms::Exception("Configuration", "Bad kind of resolution: " + skind);
    }
    float operator()(const float pt, const float abseta) const {
      for (unsigned int i = 0, n = etas.size(); i < n; ++i) {
        if (pt > ptMaxs[i])
          continue;
        if (abseta < etas[i]) {
          switch (kind) {
            case Kind::Track:
              return pt * std::min<float>(1.f, std::hypot(pt * scales[i] * 0.001, offsets[i]));
            case Kind::Calo:
              return std::min<float>(pt, pt * scales[i] + offsets[i]);
              if (pt < ptMins[i])
                return pt * std::min<float>(1, scales[i] + offsets[i] / ptMins[i]);
              return std::min<float>(pt, pt * scales[i] + offsets[i]);
          }
        }
      }
      return std::min<float>(pt, 0.3 * pt + 7);  // saturate to 100% at 10 GeV, and to 30% at high pt
    }

  protected:
    std::vector<float> etas, offsets, scales, ptMins, ptMaxs;
    enum class Kind { Calo, Track };
    Kind kind;
  };

};  // namespace l1tpf

#endif
