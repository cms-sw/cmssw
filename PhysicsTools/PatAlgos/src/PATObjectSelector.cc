#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "PhysicsTools/PatAlgos/interface/PATObjectSelector.h"

DEFINE_FWK_MODULE(pat::PATElectronSelector);
DEFINE_FWK_MODULE(pat::PATMuonSelector);
DEFINE_FWK_MODULE(pat::PATTauSelector);
DEFINE_FWK_MODULE(pat::PATJetSelector);
DEFINE_FWK_MODULE(pat::PATMETSelector);
DEFINE_FWK_MODULE(pat::PATParticleSelector);



