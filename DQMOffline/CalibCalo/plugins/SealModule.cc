#include "FWCore/Framework/interface/MakerMacros.h"

// The PhiSym and Pi0 source module
#include "DQMOffline/CalibCalo/src/DQMSourcePhiSym.h"
#include "DQMOffline/CalibCalo/src/DQMHcalPhiSymAlCaReco.h"
#include "DQMOffline/CalibCalo/src/DQMSourcePi0.h"
#include "DQMOffline/CalibCalo/interface/DQMSourceEleCalib.h"
#include "DQMOffline/CalibCalo/src/DQMHcalIsoTrackAlCaReco.cc"

DEFINE_ANOTHER_FWK_MODULE(DQMSourcePhiSym);
DEFINE_ANOTHER_FWK_MODULE(DQMHcalPhiSymAlCaReco);
DEFINE_ANOTHER_FWK_MODULE(DQMSourcePi0);
DEFINE_ANOTHER_FWK_MODULE(DQMSourceEleCalib);
DEFINE_ANOTHER_FWK_MODULE(DQMHcalIsoTrackAlCaReco);
