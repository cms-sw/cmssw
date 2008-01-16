#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/PatAlgos/interface/PATElectronProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATMuonProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATTauProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATJetProducer.h"
#include "PhysicsTools/PatAlgos/interface/PATMETProducer.h"

typedef pat::PATElectronProducer PATElectronProducer;
typedef pat::PATMuonProducer     PATMuonProducer;
typedef pat::PATTauProducer      PATTauProducer;
typedef pat::PATJetProducer      PATJetProducer;
typedef pat::PATMETProducer      PATMETProducer;

DEFINE_FWK_MODULE(PATElectronProducer);
DEFINE_FWK_MODULE(PATMuonProducer);
DEFINE_FWK_MODULE(PATTauProducer);
DEFINE_FWK_MODULE(PATJetProducer);
DEFINE_FWK_MODULE(PATMETProducer);

