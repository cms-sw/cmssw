#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerBase.h"
#include "FWCore/Utilities/interface/typelookup.h"


TYPELOOKUP_DATA_REG(EcalUnpackerWorkerBase);


