#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PatAlgos/plugins/PATJetCleaner.h"
#include "PhysicsTools/PatAlgos/plugins/PATJetCleaner.icc"

using namespace pat;
//DEFINE_FWK_MODULE(PATBaseJetCleaner);  // Gio: most likely useless
DEFINE_FWK_MODULE(PATPFJetCleaner);      
DEFINE_FWK_MODULE(PATCaloJetCleaner);


//DEFINE_FWK_MODULE(PATPF2BaseJetCleaner);
//DEFINE_FWK_MODULE(PATCalo2BaseJetCleaner);

