/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"

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

#include "CondFormats/EcalObjects/interface/EcalTPGEcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGEcalTPGCrystalStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGEcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGEcalTPGTowerStatusRcd.h"


DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(EcalPedestalsRcd,EcalPedestals);
REGISTER_PLUGIN(EcalWeightXtalGroupsRcd,EcalWeightXtalGroups);
REGISTER_PLUGIN(EcalTBWeightsRcd,EcalTBWeights);
REGISTER_PLUGIN(EcalGainRatiosRcd,EcalGainRatios);
REGISTER_PLUGIN(EcalIntercalibConstantsRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalIntercalibErrorsRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalADCToGeVConstantRcd,EcalADCToGeVConstant);
REGISTER_PLUGIN(EcalLaserAlphasRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalLaserAPDPNRatiosRcd,EcalLaserAPDPNRatios);
REGISTER_PLUGIN(EcalLaserAPDPNRatiosRefRcd,EcalFloatCondObjectContainer);
REGISTER_PLUGIN(EcalChannelStatusRcd,EcalChannelStatus);
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
