#ifndef __PhysicsTools_PatAlgos_SoftMuonMvaRun3Estimator__
#define __PhysicsTools_PatAlgos_SoftMuonMvaRun3Estimator__

#include <memory>
#include <string>

namespace pat {
  class XGBooster;
  class Muon;

  float computeSoftMvaRun3(XGBooster& booster, const Muon& muon);
}  // namespace pat
#endif
