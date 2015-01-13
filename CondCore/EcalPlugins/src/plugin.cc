/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

// #include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"


#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeDependentCorrections.h"


#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibErrorsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

// #include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"

#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"

#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"

#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"

REGISTER_PLUGIN(EcalPedestalsRcd,EcalPedestals);
REGISTER_PLUGIN(EcalWeightXtalGroupsRcd,EcalWeightXtalGroups);
REGISTER_PLUGIN(EcalTBWeightsRcd,EcalTBWeights);
REGISTER_PLUGIN(EcalGainRatiosRcd,EcalGainRatios);
REGISTER_PLUGIN(EcalLinearCorrectionsRcd,EcalLinearCorrections);
REGISTER_PLUGIN(EcalIntercalibConstantsRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalIntercalibConstantsMCRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalTimeCalibConstantsRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalTimeCalibErrorsRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalTimeOffsetConstantRcd,EcalTimeOffsetConstant);
REGISTER_PLUGIN(EcalIntercalibErrorsRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalADCToGeVConstantRcd,EcalADCToGeVConstant);
REGISTER_PLUGIN(EcalLaserAlphasRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalLaserAPDPNRatiosRcd,EcalLaserAPDPNRatios);
REGISTER_PLUGIN(EcalLaserAPDPNRatiosRefRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalChannelStatusRcd,EcalChannelStatus);

REGISTER_PLUGIN(EcalClusterCrackCorrParametersRcd,EcalFunParams);
REGISTER_PLUGIN(EcalClusterLocalContCorrParametersRcd,EcalFunParams);
REGISTER_PLUGIN(EcalClusterEnergyUncertaintyParametersRcd,EcalFunParams);
REGISTER_PLUGIN(EcalClusterEnergyCorrectionParametersRcd,EcalFunParams);
REGISTER_PLUGIN(EcalClusterEnergyCorrectionObjectSpecificParametersRcd,EcalFunParams);

REGISTER_PLUGIN(EcalMappingElectronicsRcd,EcalMappingElectronics);

REGISTER_PLUGIN(EcalTPGPedestalsRcd,EcalTPGPedestals);
REGISTER_PLUGIN(EcalTPGFineGrainEBGroupRcd,EcalTPGFineGrainEBGroup);
REGISTER_PLUGIN(EcalTPGFineGrainEBIdMapRcd,EcalTPGFineGrainEBIdMap);
REGISTER_PLUGIN(EcalTPGFineGrainStripEERcd,EcalTPGFineGrainStripEE);
REGISTER_PLUGIN(EcalTPGFineGrainTowerEERcd,EcalTPGFineGrainTowerEE);
REGISTER_PLUGIN(EcalTPGLinearizationConstRcd,EcalTPGLinearizationConst);
REGISTER_PLUGIN(EcalTPGLutGroupRcd,EcalTPGLutGroup);
REGISTER_PLUGIN(EcalTPGLutIdMapRcd,EcalTPGLutIdMap);
REGISTER_PLUGIN(EcalTPGPhysicsConstRcd,EcalTPGPhysicsConst);
REGISTER_PLUGIN(EcalTPGSlidingWindowRcd,EcalTPGSlidingWindow);
REGISTER_PLUGIN(EcalTPGWeightGroupRcd,EcalTPGWeightGroup);
REGISTER_PLUGIN(EcalTPGWeightIdMapRcd,EcalTPGWeightIdMap);
REGISTER_PLUGIN(EcalTPGCrystalStatusRcd,EcalTPGCrystalStatus);
REGISTER_PLUGIN(EcalTPGTowerStatusRcd,EcalTPGTowerStatus);
REGISTER_PLUGIN(EcalTPGStripStatusRcd,EcalTPGStripStatus);
REGISTER_PLUGIN(EcalTPGSpikeRcd,EcalTPGSpike);

REGISTER_PLUGIN(EcalDCSTowerStatusRcd,EcalDCSTowerStatus);
REGISTER_PLUGIN(EcalDAQTowerStatusRcd,EcalDAQTowerStatus);

REGISTER_PLUGIN(EcalDQMChannelStatusRcd,EcalDQMChannelStatus);
REGISTER_PLUGIN(EcalDQMTowerStatusRcd,EcalDQMTowerStatus);

REGISTER_PLUGIN(EcalSRSettingsRcd, EcalSRSettings);
REGISTER_PLUGIN(EcalSampleMaskRcd, EcalSampleMask);

REGISTER_PLUGIN(EcalTimeBiasCorrectionsRcd, EcalTimeBiasCorrections);

REGISTER_PLUGIN(EcalSamplesCorrelationRcd, EcalSamplesCorrelation);
REGISTER_PLUGIN(EcalPulseShapesRcd,EcalPulseShapes);
REGISTER_PLUGIN(EcalPulseCovariancesRcd,EcalPulseCovariances);
