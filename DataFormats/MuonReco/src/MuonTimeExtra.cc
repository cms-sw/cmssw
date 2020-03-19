#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
using namespace reco;

MuonTimeExtra::MuonTimeExtra() {
  nDof_ = 0;

  inverseBeta_ = 0.;
  inverseBetaErr_ = 0.;

  freeInverseBeta_ = 0.;
  freeInverseBetaErr_ = 0.;

  timeAtIpInOut_ = 0.;
  timeAtIpInOutErr_ = 0.;
  timeAtIpOutIn_ = 0.;
  timeAtIpOutInErr_ = 0.;
}
