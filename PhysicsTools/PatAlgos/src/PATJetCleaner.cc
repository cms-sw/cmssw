#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PatAlgos/interface/PATJetCleaner.h"
#include "PhysicsTools/PatAlgos/interface/PATJetCleaner.icc"

DEFINE_FWK_MODULE(pat::PATBaseJetCleaner);
DEFINE_FWK_MODULE(pat::PATPFJetCleaner);
DEFINE_FWK_MODULE(pat::PATCaloJetCleaner);
//DEFINE_FWK_MODULE(pat::PATPF2BaseJetCleaner);
//DEFINE_FWK_MODULE(pat::PATCalo2BaseJetCleaner);

