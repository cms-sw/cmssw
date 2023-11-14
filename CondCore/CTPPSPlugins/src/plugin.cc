#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"
#include "CondFormats/AlignmentRecord/interface/RPMisalignedAlignmentRecord.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSOpticsRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelGainCalibrationsRcd.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigRcd.h"
#include "CondFormats/DataRecord/interface/PPSAlignmentConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/PPSAssociationCutsRcd.h"
#include "CondFormats/DataRecord/interface/PPSDirectSimulationDataRcd.h"
#include "CondFormats/DataRecord/interface/PPSPixelTopologyRcd.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationLUTRcd.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"
#include "CondFormats/DataRecord/interface/TotemAnalysisMaskRcd.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelGainCalibrations.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondFormats/PPSObjects/interface/LHCOpticalFunctionsSetCollection.h"
#include "CondFormats/PPSObjects/interface/PPSAlignmentConfig.h"
#include "CondFormats/PPSObjects/interface/PPSAlignmentConfiguration.h"
#include "CondFormats/PPSObjects/interface/PPSAssociationCuts.h"
#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibrationLUT.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"

namespace {
  struct InitAssociationCuts {
    void operator()(PPSAssociationCuts &cuts) { cuts.initialize(); }
  };
}  // namespace

REGISTER_PLUGIN(CTPPSBeamParametersRcd, CTPPSBeamParameters);
REGISTER_PLUGIN(CTPPSPixelDAQMappingRcd, CTPPSPixelDAQMapping);
REGISTER_PLUGIN(CTPPSPixelAnalysisMaskRcd, CTPPSPixelAnalysisMask);
REGISTER_PLUGIN(CTPPSPixelGainCalibrationsRcd, CTPPSPixelGainCalibrations);
REGISTER_PLUGIN(CTPPSRPAlignmentCorrectionsDataRcd, CTPPSRPAlignmentCorrectionsData);
REGISTER_PLUGIN_NO_SERIAL(RPRealAlignmentRecord, CTPPSRPAlignmentCorrectionsData);
REGISTER_PLUGIN_NO_SERIAL(RPMisalignedAlignmentRecord, CTPPSRPAlignmentCorrectionsData);
REGISTER_PLUGIN(PPSTimingCalibrationRcd, PPSTimingCalibration);
REGISTER_PLUGIN(PPSTimingCalibrationLUTRcd, PPSTimingCalibrationLUT);
REGISTER_PLUGIN(CTPPSOpticsRcd, LHCOpticalFunctionsSetCollection);
REGISTER_PLUGIN(PPSDirectSimulationDataRcd, PPSDirectSimulationData);
REGISTER_PLUGIN(PPSPixelTopologyRcd, PPSPixelTopology);
REGISTER_PLUGIN(PPSAlignmentConfigRcd, PPSAlignmentConfig);
REGISTER_PLUGIN(PPSAlignmentConfigurationRcd, PPSAlignmentConfiguration);
REGISTER_PLUGIN(TotemAnalysisMaskRcd, TotemAnalysisMask);
REGISTER_PLUGIN(TotemReadoutRcd, TotemDAQMapping);

REGISTER_PLUGIN_INIT(PPSAssociationCutsRcd, PPSAssociationCuts, InitAssociationCuts);
