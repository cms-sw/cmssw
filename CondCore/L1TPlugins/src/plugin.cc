#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondTools/L1Trigger/interface/WriterProxy.h"

#include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

// Central L1 records
REGISTER_PLUGIN(L1TriggerKeyRcd, L1TriggerKey);
REGISTER_L1_WRITER(L1TriggerKeyRcd, L1TriggerKey);

// L1 scales
REGISTER_PLUGIN(L1JetEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1JetEtScaleRcd, L1CaloEtScale);

REGISTER_PLUGIN(L1EmEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1EmEtScaleRcd, L1CaloEtScale);

REGISTER_PLUGIN(L1MuTriggerScalesRcd, L1MuTriggerScales);
REGISTER_L1_WRITER(L1MuTriggerScalesRcd, L1MuTriggerScales);

REGISTER_PLUGIN(L1MuGMTScalesRcd, L1MuGMTScales);
REGISTER_L1_WRITER(L1MuGMTScalesRcd, L1MuGMTScales);

// DT TPG records
REGISTER_PLUGIN(DTConfigManagerRcd, DTConfigManager);
REGISTER_L1_WRITER(DTConfigManagerRcd, DTConfigManager);

// DT TF records
//REGISTER_PLUGIN();
//REGISTER_L1_WRITER();

// CSC TPG records
REGISTER_PLUGIN(L1CSCTPParametersRcd, L1CSCTPParameters);
REGISTER_L1_WRITER(L1CSCTPParametersRcd, L1CSCTPParameters);

// CSC TF records
//REGISTER_PLUGIN();
//REGISTER_L1_WRITER();

// GMT records
REGISTER_PLUGIN(L1MuGMTParametersRcd, L1MuGMTParameters);
REGISTER_L1_WRITER(L1MuGMTParametersRcd, L1MuGMTParameters);

// RCT records
//REGISTER_PLUGIN(L1RCTParametersRcd);
//REGISTER_L1_WRITER(L1RCTParametersRcd);

// GCT records
REGISTER_PLUGIN(L1GctJetFinderParamsRcd, L1GctJetFinderParams);
REGISTER_L1_WRITER(L1GctJetFinderParamsRcd, L1GctJetFinderParams);

REGISTER_PLUGIN(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);
REGISTER_L1_WRITER(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);

REGISTER_PLUGIN(L1GctJetCounterNegativeEtaRcd, L1GctJetCounterSetup);
REGISTER_L1_WRITER(L1GctJetCounterNegativeEtaRcd, L1GctJetCounterSetup);

REGISTER_PLUGIN(L1GctJetCounterPositiveEtaRcd, L1GctJetCounterSetup);
REGISTER_L1_WRITER(L1GctJetCounterPositiveEtaRcd, L1GctJetCounterSetup);



// GT records
REGISTER_PLUGIN(L1GtBoardMapsRcd, L1GtBoardMaps);
REGISTER_L1_WRITER(L1GtBoardMapsRcd, L1GtBoardMaps);

REGISTER_PLUGIN(L1GtParametersRcd, L1GtParameters);
REGISTER_L1_WRITER(L1GtParametersRcd, L1GtParameters);

REGISTER_PLUGIN(L1GtPrescaleFactorsRcd, L1GtPrescaleFactors);
REGISTER_L1_WRITER(L1GtPrescaleFactorsRcd, L1GtPrescaleFactors);

REGISTER_PLUGIN(L1GtStableParametersRcd, L1GtStableParameters);
REGISTER_L1_WRITER(L1GtStableParametersRcd, L1GtStableParameters);

REGISTER_PLUGIN(L1GtTriggerMaskRcd, L1GtTriggerMask);
REGISTER_L1_WRITER(L1GtTriggerMaskRcd, L1GtTriggerMask);

REGISTER_PLUGIN(L1GtTriggerMenuRcd, L1GtTriggerMenu);
REGISTER_L1_WRITER(L1GtTriggerMenuRcd, L1GtTriggerMenu);
