#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PatAlgos/plugins/PATJetCleaner.h"
#include "PhysicsTools/PatAlgos/plugins/PATJetCleaner.icc"

using namespace pat;
DEFINE_FWK_MODULE(PATBasicJetCleaner);
DEFINE_FWK_MODULE(PATCaloJetCleaner);
DEFINE_FWK_MODULE(PATPFJetCleaner);      

