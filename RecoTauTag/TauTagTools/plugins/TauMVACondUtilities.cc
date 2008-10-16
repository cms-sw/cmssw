#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"

using namespace PhysicsTools;

typedef MVAComputerESSourceImpl<BTauGenericMVAJetTagComputerRcd> TauMVAComputerESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(TauMVAComputerESSource);

typedef MVATrainerContainerSaveImpl<BTauGenericMVAJetTagComputerRcd> TauMVATrainerSave;
DEFINE_FWK_MODULE(TauMVATrainerSave);


