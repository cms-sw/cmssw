#include "PhysicsTools/PatAlgos/plugins/PATCleaner.h"

#include "FWCore/Framework/interface/MakerMacros.h"
namespace pat {
    typedef pat::PATCleaner<pat::Electron>   PATElectronCleaner;
    typedef pat::PATCleaner<pat::Muon>       PATMuonCleaner;
    typedef pat::PATCleaner<pat::Tau>        PATTauCleaner;
    typedef pat::PATCleaner<pat::Photon>     PATPhotonCleaner;
    typedef pat::PATCleaner<pat::Jet>        PATJetCleaner;
    typedef pat::PATCleaner<pat::MET>        PATMETCleaner;
    typedef pat::PATCleaner<pat::GenericParticle> PATGenericParticleCleaner;
    typedef pat::PATCleaner<pat::PFParticle> PATPFParticleCleaner; 
}
using namespace pat;
DEFINE_FWK_MODULE(PATElectronCleaner);
DEFINE_FWK_MODULE(PATMuonCleaner);
DEFINE_FWK_MODULE(PATTauCleaner);
DEFINE_FWK_MODULE(PATPhotonCleaner);
DEFINE_FWK_MODULE(PATJetCleaner);
DEFINE_FWK_MODULE(PATMETCleaner);
DEFINE_FWK_MODULE(PATGenericParticleCleaner);
DEFINE_FWK_MODULE(PATPFParticleCleaner);
