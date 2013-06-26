#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooperImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerFileSaveImpl.h"
#include "JetTagMVATrainer.h"

// the main module
DEFINE_FWK_MODULE(JetTagMVATrainer);

// trainer helpers
using namespace PhysicsTools;

typedef MVATrainerContainerLooperImpl<BTauGenericMVAJetTagComputerRcd> JetTagMVATrainerLooper;
DEFINE_FWK_LOOPER(JetTagMVATrainerLooper);

typedef MVATrainerContainerSaveImpl<BTauGenericMVAJetTagComputerRcd> JetTagMVATrainerSave;
DEFINE_FWK_MODULE(JetTagMVATrainerSave);

typedef MVATrainerFileSaveImpl<BTauGenericMVAJetTagComputerRcd> JetTagMVATrainerFileSave;
DEFINE_FWK_MODULE(JetTagMVATrainerFileSave);

typedef MVAComputerESSourceImpl<BTauGenericMVAJetTagComputerRcd> BTauGenericMVAJetTagComputerFileSource;
DEFINE_FWK_EVENTSETUP_SOURCE(BTauGenericMVAJetTagComputerFileSource);
