#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Calibration/HcalIsolatedTrackReco/interface/IsolatedPixelTrackCandidateProducer.h"
#include "Calibration/HcalIsolatedTrackReco/interface/RegionalEcalClusterProducer.h"
#include "Calibration/HcalIsolatedTrackReco/interface/EcalIsolatedParticleCandidateProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(IsolatedPixelTrackCandidateProducer);
DEFINE_ANOTHER_FWK_MODULE(RegionalEcalClusterProducer);
DEFINE_ANOTHER_FWK_MODULE(EcalIsolatedParticleCandidateProducer);
