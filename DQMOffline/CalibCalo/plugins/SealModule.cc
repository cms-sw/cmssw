#include "FWCore/Framework/interface/MakerMacros.h"

// The PhiSym and Pi0 source module
#include "DQMOffline/CalibCalo/src/DQMSourcePhiSym.h"
#include "DQMOffline/CalibCalo/src/DQMSourcePi0.h"
#include "DQMOffline/CalibCalo/interface/DQMSourceEleCalib.h"

DEFINE_ANOTHER_FWK_MODULE(DQMSourcePhiSym);
DEFINE_ANOTHER_FWK_MODULE(DQMSourcePi0);
DEFINE_ANOTHER_FWK_MODULE(DQMSourceEleCalib);
