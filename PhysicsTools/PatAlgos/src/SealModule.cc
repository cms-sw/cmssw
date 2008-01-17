#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATJetProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATMETProducer.h"

using namespace pat;

DEFINE_FWK_MODULE(PATElectronProducer);
DEFINE_FWK_MODULE(PATMuonProducer);
DEFINE_FWK_MODULE(PATTauProducer);
DEFINE_FWK_MODULE(PATJetProducer);
DEFINE_FWK_MODULE(PATMETProducer);

