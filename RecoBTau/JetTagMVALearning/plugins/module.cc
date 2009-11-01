#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooperImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"
#include "RecoBTau/JetTagMVALearning/interface/JetTagMVATrainer.h"

// the main module
DEFINE_FWK_MODULE(JetTagMVATrainer);

// trainer helpers
using namespace PhysicsTools;

typedef MVATrainerContainerLooperImpl<BTauGenericMVAJetTagComputerRcd> JetTagMVATrainerLooper;
DEFINE_FWK_LOOPER(JetTagMVATrainerLooper);

typedef MVATrainerContainerSaveImpl<BTauGenericMVAJetTagComputerRcd> JetTagMVATrainerSave;
DEFINE_FWK_MODULE(JetTagMVATrainerSave);
