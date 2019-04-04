#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelGainCalibrationsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

REGISTER_PLUGIN(CTPPSBeamParametersRcd,CTPPSBeamParameters);
REGISTER_PLUGIN(CTPPSPixelDAQMappingRcd,CTPPSPixelDAQMapping);
REGISTER_PLUGIN(CTPPSPixelAnalysisMaskRcd,CTPPSPixelAnalysisMask);
REGISTER_PLUGIN(CTPPSPixelGainCalibrationsRcd,CTPPSPixelGainCalibrations);
REGISTER_PLUGIN(CTPPSRPAlignmentCorrectionsDataRcd,CTPPSRPAlignmentCorrectionsData);
REGISTER_PLUGIN(RPRealAlignmentRecord,CTPPSRPAlignmentCorrectionsData);
REGISTER_PLUGIN(RPMisalignedAlignmentRecord,CTPPSRPAlignmentCorrectionsData);
REGISTER_PLUGIN(PPSTimingCalibrationRcd,PPSTimingCalibration);

