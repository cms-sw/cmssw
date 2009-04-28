#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerBase.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"


EVENTSETUP_DATA_REG(EcalUnpackerWorkerBase);


