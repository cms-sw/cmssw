#ifndef __PhysicsTools_PatAlgos_SoftMuonMvaRun3Estimator__
#define __PhysicsTools_PatAlgos_SoftMuonMvaRun3Estimator__

#include <memory>
#include <string>
#include "DataFormats/PatCandidates/interface/MuonFwd.h"

namespace pat {
  class XGBooster;

  float computeSoftMvaRun3(XGBooster& booster, const Muon& muon);
}  // namespace pat
#endif
