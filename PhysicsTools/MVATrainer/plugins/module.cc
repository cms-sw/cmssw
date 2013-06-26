#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerRecord.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerLooperImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"

using namespace PhysicsTools;

// define trainer modules using the dummy MVAComputer record for testing purposes

typedef MVATrainerContainerLooperImpl<MVAComputerRecord> MVAComputerTrainerLooper;
DEFINE_FWK_LOOPER(MVAComputerTrainerLooper);

typedef MVATrainerContainerSaveImpl<MVAComputerRecord> MVAComputerTrainerSave;
DEFINE_FWK_MODULE(MVAComputerTrainerSave);
