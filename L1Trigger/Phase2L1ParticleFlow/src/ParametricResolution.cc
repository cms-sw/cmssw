#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"

std::vector<float> l1tpf::ParametricResolution::getVFloat(const edm::ParameterSet &cpset, const std::string &name) {
  std::vector<double> vd = cpset.getParameter<std::vector<double>>(name);
  return std::vector<float>(vd.begin(), vd.end());
}

l1tpf::ParametricResolution::ParametricResolution(const edm::ParameterSet &cpset)
    : etas_(getVFloat(cpset, "etaBins")), offsets_(getVFloat(cpset, "offset")), scales_(getVFloat(cpset, "scale")) {
  if (cpset.existsAs<std::vector<double>>("ptMin")) {
    ptMins_ = getVFloat(cpset, "ptMin");
  } else {
    float ptMin = cpset.existsAs<double>("ptMin") ? cpset.getParameter<double>("ptMin") : 0;
    ptMins_ = std::vector<float>(etas_.size(), ptMin);
  }
  if (cpset.existsAs<std::vector<double>>("ptMax")) {
    ptMaxs_ = getVFloat(cpset, "ptMax");
  } else {
    ptMaxs_ = std::vector<float>(etas_.size(), 1e6);
  }

  std::string skind = cpset.getParameter<std::string>("kind");
  if (skind == "track")
    kind_ = Kind::Track;
  else if (skind == "calo")
    kind_ = Kind::Calo;
  else
    throw cms::Exception("Configuration", "Bad kind of resolution: " + skind);
}

float l1tpf::ParametricResolution::operator()(const float pt, const float abseta) const {
  for (unsigned int i = 0, n = etas_.size(); i < n; ++i) {
    if (pt > ptMaxs_[i])
      continue;
    if (abseta < etas_[i]) {
      switch (kind_) {
        case Kind::Track:
          return pt * std::min<float>(1.f, std::hypot(pt * scales_[i] * 0.001, offsets_[i]));
        case Kind::Calo:
          return std::min<float>(pt, pt * scales_[i] + offsets_[i]);
          if (pt < ptMins_[i])
            return pt * std::min<float>(1, scales_[i] + offsets_[i] / ptMins_[i]);
          return std::min<float>(pt, pt * scales_[i] + offsets_[i]);
      }
    }
  }
  return std::min<float>(pt, 0.3 * pt + 7);  // saturate to 100% at 10 GeV, and to 30% at high pt
}
