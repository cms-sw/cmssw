#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "RecoTauTag/TauTagTools/interface/TauMVADBConfiguration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"

using namespace PhysicsTools;

typedef MVAComputerESSourceImpl<TauMVAFrameworkDBRcd> TauMVAComputerESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(TauMVAComputerESSource);

typedef MVATrainerContainerSaveImpl<TauMVAFrameworkDBRcd> TauMVATrainerSave;
DEFINE_FWK_MODULE(TauMVATrainerSave);


