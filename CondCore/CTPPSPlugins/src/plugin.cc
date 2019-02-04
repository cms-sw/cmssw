#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelGainCalibrationsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

REGISTER_PLUGIN(CTPPSPixelDAQMappingRcd,CTPPSPixelDAQMapping);
REGISTER_PLUGIN(CTPPSPixelAnalysisMaskRcd,CTPPSPixelAnalysisMask);
REGISTER_PLUGIN(CTPPSPixelGainCalibrationsRcd,CTPPSPixelGainCalibrations);
REGISTER_PLUGIN(PPSTimingCalibrationRcd,PPSTimingCalibration);

