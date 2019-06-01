#include "FWCore/Framework/interface/MakerMacros.h"

// The PhiSym and Pi0 source module
#include "DQMOffline/CalibCalo/interface/DQMSourceEleCalib.h"
#include "DQMOffline/CalibCalo/src/DQMHOAlCaRecoStream.h"
#include "DQMOffline/CalibCalo/src/DQMHcalDiJetsAlCaReco.h"
#include "DQMOffline/CalibCalo/src/DQMHcalIsoTrackAlCaReco.h"
#include "DQMOffline/CalibCalo/src/DQMHcalIsolatedBunchAlCaReco.h"
#include "DQMOffline/CalibCalo/src/DQMHcalPhiSymAlCaReco.h"
#include "DQMOffline/CalibCalo/src/DQMSourcePi0.h"

DEFINE_FWK_MODULE(DQMHcalPhiSymAlCaReco);
DEFINE_FWK_MODULE(DQMSourcePi0);
DEFINE_FWK_MODULE(DQMSourceEleCalib);
DEFINE_FWK_MODULE(DQMHcalIsoTrackAlCaReco);
DEFINE_FWK_MODULE(DQMHcalDiJetsAlCaReco);
DEFINE_FWK_MODULE(DQMHOAlCaRecoStream);
DEFINE_FWK_MODULE(DQMHcalIsolatedBunchAlCaReco);
