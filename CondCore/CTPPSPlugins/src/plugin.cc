#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"

 
REGISTER_PLUGIN(CTPPSPixelDAQMappingRcd,CTPPSPixelDAQMapping);
REGISTER_PLUGIN(CTPPSPixelAnalysisMaskRcd,CTPPSPixelAnalysisMask);
