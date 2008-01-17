#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "PhysicsTools/PatAlgos/interface/PATObjectSelector.h"

using namespace pat;

DEFINE_FWK_MODULE(PATElectronSelector);
DEFINE_FWK_MODULE(PATMuonSelector);
DEFINE_FWK_MODULE(PATTauSelector);
DEFINE_FWK_MODULE(PATJetSelector);
DEFINE_FWK_MODULE(PATMETSelector);
DEFINE_FWK_MODULE(PATParticleSelector);



