#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/HcalRawToDigi/plugins/HcalRawToDigi.h"
#include "EventFilter/HcalRawToDigi/plugins/HcalDigiToRaw.h"
#include "EventFilter/HcalRawToDigi/plugins/HcalHistogramRawToDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalRawToDigi);
DEFINE_ANOTHER_FWK_MODULE(HcalHistogramRawToDigi);
DEFINE_ANOTHER_FWK_MODULE(HcalDigiToRaw);
