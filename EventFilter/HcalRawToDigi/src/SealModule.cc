#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/HcalRawToDigi/interface/HcalRawToDigi.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHistogramRawToDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalRawToDigi)
DEFINE_ANOTHER_FWK_MODULE(HcalHistogramRawToDigi)

