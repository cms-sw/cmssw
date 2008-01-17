#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PatAlgos/interface/PATJetCleaner.h"
#include "PhysicsTools/PatAlgos/interface/PATJetCleaner.icc"

using namespace pat;
DEFINE_FWK_MODULE(PATBaseJetCleaner);
DEFINE_FWK_MODULE(PATPFJetCleaner);
DEFINE_FWK_MODULE(PATCaloJetCleaner);
//DEFINE_FWK_MODULE(PATPF2BaseJetCleaner);
//DEFINE_FWK_MODULE(PATCalo2BaseJetCleaner);

