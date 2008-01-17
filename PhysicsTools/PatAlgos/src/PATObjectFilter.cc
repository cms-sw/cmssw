
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "PhysicsTools/PatAlgos/interface/PATObjectFilter.h"

using namespace pat;

DEFINE_FWK_MODULE(PATElectronMinFilter);
DEFINE_FWK_MODULE(PATMuonMinFilter);
DEFINE_FWK_MODULE(PATTauMinFilter);
DEFINE_FWK_MODULE(PATJetMinFilter);
DEFINE_FWK_MODULE(PATMETMinFilter);
DEFINE_FWK_MODULE(PATParticleMinFilter);

DEFINE_FWK_MODULE(PATElectronMaxFilter);
DEFINE_FWK_MODULE(PATMuonMaxFilter);
DEFINE_FWK_MODULE(PATTauMaxFilter);
DEFINE_FWK_MODULE(PATJetMaxFilter);
DEFINE_FWK_MODULE(PATMETMaxFilter);
DEFINE_FWK_MODULE(PATParticleMaxFilter);


