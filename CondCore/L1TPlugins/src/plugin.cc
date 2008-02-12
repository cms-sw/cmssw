#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondTools/L1Trigger/interface/WriterProxy.h"

DEFINE_SEAL_MODULE();

// // Central L1 records
// #include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
// #include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

// REGISTER_PLUGIN(L1TriggerKeyRcd, L1TriggerKey);
// REGISTER_L1_WRITER(L1TriggerKeyRcd, L1TriggerKey);

// // L1 scales
// #include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
// #include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
// #include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

// REGISTER_PLUGIN(L1JetEtScaleRcd, L1CaloEtScale);
// REGISTER_L1_WRITER(L1JetEtScaleRcd, L1CaloEtScale);

// REGISTER_PLUGIN(L1EmEtScaleRcd, L1CaloEtScale);
// REGISTER_L1_WRITER(L1EmEtScaleRcd, L1CaloEtScale);

// #include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
// #include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"

// REGISTER_PLUGIN(L1MuTriggerScalesRcd, L1MuTriggerScales);
// REGISTER_L1_WRITER(L1MuTriggerScalesRcd, L1MuTriggerScales);

// #include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
// #include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"

// REGISTER_PLUGIN(L1MuGMTScalesRcd, L1MuGMTScales);
// REGISTER_L1_WRITER(L1MuGMTScalesRcd, L1MuGMTScales);

// // DT TPG records
// #include "CondFormats/L1TObjects/interface/DTConfigManager.h"
// #include "CondFormats/DataRecord/interface/DTConfigManagerRcd.h"

// REGISTER_PLUGIN(DTConfigManagerRcd, DTConfigManager);
// REGISTER_L1_WRITER(DTConfigManagerRcd, DTConfigManager);

// // DT TF records
// //#include "CondFormats/L1TObjects/interface/"
// //#include "CondFormats/DataRecord/interface/"

// //REGISTER_PLUGIN();
// //REGISTER_L1_WRITER();

// // CSC TPG records
// #include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"
// #include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"

// REGISTER_PLUGIN(L1CSCTPParametersRcd, L1CSCTPParameters);
// REGISTER_L1_WRITER(L1CSCTPParametersRcd, L1CSCTPParameters);

// // CSC TF records
// //#include "CondFormats/L1TObjects/interface/"
// //#include "CondFormats/DataRecord/interface/"

// //REGISTER_PLUGIN();
// //REGISTER_L1_WRITER();

// // GMT records
// #include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
// #include "CondFormats/DataRecord/interface/L1MuGMTParametersRcd.h"

// REGISTER_PLUGIN(L1MuGMTParametersRcd, L1MuGMTParameters);
// REGISTER_L1_WRITER(L1MuGMTParametersRcd, L1MuGMTParameters);

// // RCT records
// #include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
// #include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

// REGISTER_PLUGIN(L1RCTParametersRcd, L1RCTParameters);
// REGISTER_L1_WRITER(L1RCTParametersRcd, L1RCTParameters);

// // GCT records
// #include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
// #include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

// REGISTER_PLUGIN(L1GctJetFinderParamsRcd, L1GctJetFinderParams);
// REGISTER_L1_WRITER(L1GctJetFinderParamsRcd, L1GctJetFinderParams);

// #include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
// #include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"

// REGISTER_PLUGIN(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);
// REGISTER_L1_WRITER(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);

// #include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"
// #include "CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h"
// #include "CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h"

// REGISTER_PLUGIN(L1GctJetCounterNegativeEtaRcd, L1GctJetCounterSetup);
// REGISTER_L1_WRITER(L1GctJetCounterNegativeEtaRcd, L1GctJetCounterSetup);

// REGISTER_PLUGIN(L1GctJetCounterPositiveEtaRcd, L1GctJetCounterSetup);
// REGISTER_L1_WRITER(L1GctJetCounterPositiveEtaRcd, L1GctJetCounterSetup);



// // GT records
// #include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
// #include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

// REGISTER_PLUGIN(L1GtBoardMapsRcd, L1GtBoardMaps);
// REGISTER_L1_WRITER(L1GtBoardMapsRcd, L1GtBoardMaps);

// #include "CondFormats/L1TObjects/interface/L1GtParameters.h"
// #include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

// REGISTER_PLUGIN(L1GtParametersRcd, L1GtParameters);
// REGISTER_L1_WRITER(L1GtParametersRcd, L1GtParameters);

// #include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
// #include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsRcd.h"

// REGISTER_PLUGIN(L1GtPrescaleFactorsRcd, L1GtPrescaleFactors);
// REGISTER_L1_WRITER(L1GtPrescaleFactorsRcd, L1GtPrescaleFactors);

// #include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
// #include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

// REGISTER_PLUGIN(L1GtStableParametersRcd, L1GtStableParameters);
// REGISTER_L1_WRITER(L1GtStableParametersRcd, L1GtStableParameters);

// #include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
// #include "CondFormats/DataRecord/interface/L1GtTriggerMaskRcd.h"

// REGISTER_PLUGIN(L1GtTriggerMaskRcd, L1GtTriggerMask);
// REGISTER_L1_WRITER(L1GtTriggerMaskRcd, L1GtTriggerMask);

// #include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
// #include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// REGISTER_PLUGIN(L1GtTriggerMenuRcd, L1GtTriggerMenu);
// REGISTER_L1_WRITER(L1GtTriggerMenuRcd, L1GtTriggerMenu);
