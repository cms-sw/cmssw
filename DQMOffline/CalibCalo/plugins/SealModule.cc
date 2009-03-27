#include "FWCore/Framework/interface/MakerMacros.h"

// The PhiSym and Pi0 source module
#include "DQMOffline/CalibCalo/src/DQMSourcePhiSym.h"
#include "DQMOffline/CalibCalo/src/DQMHcalPhiSymAlCaReco.h"
#include "DQMOffline/CalibCalo/src/DQMSourcePi0.h"
#include "DQMOffline/CalibCalo/interface/DQMSourceEleCalib.h"
#include "DQMOffline/CalibCalo/src/DQMHcalIsoTrackAlCaReco.cc"
#include "DQMOffline/CalibCalo/src/DQMHcalDiJetsAlCaReco.h"
#include "DQMOffline/CalibCalo/src/DQMHcalDiJetsAlCaReco.cc"
#include "DQMOffline/CalibCalo/src/DQMEcalCalibConstants.h"

DEFINE_ANOTHER_FWK_MODULE(DQMSourcePhiSym);
DEFINE_ANOTHER_FWK_MODULE(DQMHcalPhiSymAlCaReco);
DEFINE_ANOTHER_FWK_MODULE(DQMSourcePi0);
DEFINE_ANOTHER_FWK_MODULE(DQMSourceEleCalib);
DEFINE_ANOTHER_FWK_MODULE(DQMHcalIsoTrackAlCaReco);
DEFINE_ANOTHER_FWK_MODULE(DQMHcalDiJetsAlCaReco);
DEFINE_ANOTHER_FWK_MODULE(DQMEcalCalibConstants);
