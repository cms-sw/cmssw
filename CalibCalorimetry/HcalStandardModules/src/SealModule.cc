#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <CalibCalorimetry/HcalStandardModules/interface/HcalPedestalAnalysis.h>
//#include "EventFilter/HcalRawToDigi/interface/HcalHistogramRawToDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalPedestalAnalysis)
