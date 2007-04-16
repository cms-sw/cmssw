#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/HcalRawToDigi/src/HcalRawToDigi.h"
#include "EventFilter/HcalRawToDigi/src/HcalDigiToRaw.h"
#include "EventFilter/HcalRawToDigi/src/HcalHistogramRawToDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalRawToDigi);
DEFINE_ANOTHER_FWK_MODULE(HcalHistogramRawToDigi);
DEFINE_ANOTHER_FWK_MODULE(HcalDigiToRaw);
