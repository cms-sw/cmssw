
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "PhysicsTools/PatAlgos/interface/PATObjectFilter.h"

DEFINE_FWK_MODULE(pat::PATElectronMinFilter);
DEFINE_FWK_MODULE(pat::PATMuonMinFilter);
DEFINE_FWK_MODULE(pat::PATTauMinFilter);
DEFINE_FWK_MODULE(pat::PATJetMinFilter);
DEFINE_FWK_MODULE(pat::PATMETMinFilter);
DEFINE_FWK_MODULE(pat::PATParticleMinFilter);

DEFINE_FWK_MODULE(pat::PATElectronMaxFilter);
DEFINE_FWK_MODULE(pat::PATMuonMaxFilter);
DEFINE_FWK_MODULE(pat::PATTauMaxFilter);
DEFINE_FWK_MODULE(pat::PATJetMaxFilter);
DEFINE_FWK_MODULE(pat::PATMETMaxFilter);
DEFINE_FWK_MODULE(pat::PATParticleMaxFilter);


