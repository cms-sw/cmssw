#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/LooperFactory.h"

#include "RecoTauTag/TauTagTools/interface/TauMVADBConfiguration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooperImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerFileSaveImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"

using namespace PhysicsTools;

typedef MVATrainerContainerLooperImpl<TauMVAFrameworkDBRcd> TauMVATrainerLooper;
DEFINE_FWK_LOOPER(TauMVATrainerLooper);

typedef MVAComputerESSourceImpl<TauMVAFrameworkDBRcd> TauMVAComputerESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(TauMVAComputerESSource);

typedef MVATrainerContainerSaveImpl<TauMVAFrameworkDBRcd> TauMVATrainerSave;
DEFINE_FWK_MODULE(TauMVATrainerSave);

typedef MVATrainerFileSaveImpl<TauMVAFrameworkDBRcd> TauMVATrainerFileSave;
DEFINE_FWK_MODULE(TauMVATrainerFileSave);
