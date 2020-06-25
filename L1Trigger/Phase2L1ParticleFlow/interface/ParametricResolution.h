#ifndef L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h
#define L1Trigger_Phase2L1ParticleFlow_ParametricResolution_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <vector>
#include <cmath>

namespace l1tpf {
  class ParametricResolution {
  public:
    ParametricResolution() {}
    ParametricResolution(const edm::ParameterSet &cpset) {
      std::vector<double> etaBins = cpset.getParameter<std::vector<double>>("etaBins");
      std::vector<double> offset = cpset.getParameter<std::vector<double>>("offset");
      std::vector<double> scale = cpset.getParameter<std::vector<double>>("scale");
      etas.insert(etas.end(), etaBins.begin(), etaBins.end());
      scales.insert(scales.end(), scale.begin(), scale.end());
      offsets.insert(offsets.end(), offset.begin(), offset.end());
      if (cpset.existsAs<std::vector<double>>("ptMin")) {
        std::vector<double> ptMin = cpset.getParameter<std::vector<double>>("ptMin");
        ptMins.insert(ptMins.end(), ptMin.begin(), ptMin.end());
      } else {
        float ptMin = cpset.existsAs<double>("ptMin") ? cpset.getParameter<double>("ptMin") : 0;
        ptMins = std::vector<float>(etaBins.size(), ptMin);
      }
      if (cpset.existsAs<std::vector<double>>("ptMax")) {
        std::vector<double> ptMax = cpset.getParameter<std::vector<double>>("ptMax");
        ptMaxs.insert(ptMaxs.end(), ptMax.begin(), ptMax.end());
      } else {
        ptMaxs = std::vector<float>(etaBins.size(), 1e6);
      }
      std::string skind = cpset.getParameter<std::string>("kind");
      if (skind == "track")
        kind = Kind::Track;
      else if (skind == "calo")
        kind = Kind::Calo;
      else
        throw cms::Exception("Configuration", "Bad kind of resolution: "+skind);
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
