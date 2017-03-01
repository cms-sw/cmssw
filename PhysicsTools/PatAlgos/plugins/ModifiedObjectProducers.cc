#include "ModifiedObjectProducer.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

typedef pat::ModifiedObjectProducer<pat::Electron> ModifiedElectronProducer;
typedef pat::ModifiedObjectProducer<pat::Photon>   ModifiedPhotonProducer;
typedef pat::ModifiedObjectProducer<pat::Muon>     ModifiedMuonProducer;
typedef pat::ModifiedObjectProducer<pat::Tau>      ModifiedTauProducer;
typedef pat::ModifiedObjectProducer<pat::Jet>      ModifiedJetProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ModifiedElectronProducer);
DEFINE_FWK_MODULE(ModifiedPhotonProducer);
DEFINE_FWK_MODULE(ModifiedMuonProducer);
DEFINE_FWK_MODULE(ModifiedTauProducer);
DEFINE_FWK_MODULE(ModifiedJetProducer);
