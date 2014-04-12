#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "EventFilter/EcalRawToDigi/plugins/EcalRegionCablingESProducer.h"
DEFINE_FWK_EVENTSETUP_MODULE(EcalRegionCablingESProducer);

#include "EventFilter/EcalRawToDigi/plugins/EcalRawToDigi.h"
DEFINE_FWK_MODULE(EcalRawToDigi);

#include "EventFilter/EcalRawToDigi/interface/MatacqProducer.h"
DEFINE_FWK_MODULE(MatacqProducer);

#include "EventFilter/EcalRawToDigi/interface/EcalDumpRaw.h"
DEFINE_FWK_MODULE(EcalDumpRaw);

