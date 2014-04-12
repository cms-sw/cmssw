#include "FWCore/Framework/interface/MakerMacros.h"


#include "EventFilter/SiStripRawToDigi/plugins/SiStripDigiToRawModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToDigiModule.h"
#include "EventFilter/SiStripRawToDigi/plugins/ExcludedFEDListProducer.h"

typedef sistrip::DigiToRawModule SiStripDigiToRawModule;
typedef sistrip::RawToDigiModule SiStripRawToDigiModule;
typedef sistrip::ExcludedFEDListProducer SiStripExcludedFEDListProducer;

DEFINE_FWK_MODULE(SiStripDigiToRawModule);
DEFINE_FWK_MODULE(SiStripRawToDigiModule);
DEFINE_FWK_MODULE(SiStripExcludedFEDListProducer);

