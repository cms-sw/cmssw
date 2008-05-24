#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/L1Trigger/plugins/L1CondDBPayloadWriter.h"
#include "CondTools/L1Trigger/plugins/L1CondDBIOVWriter.h"
#include "CondTools/L1Trigger/plugins/L1TriggerKeyDummyProd.h"
#include "CondTools/L1Trigger/plugins/L1TriggerKeyListDummyProd.h"
#include "CondTools/L1Trigger/plugins/L1TriggerKeyOnlineProd.h"
#include "CondTools/L1Trigger/plugins/L1TriggerConfigOnlineProd.h"

using namespace l1t;

DEFINE_FWK_MODULE(L1CondDBPayloadWriter);
DEFINE_ANOTHER_FWK_MODULE(L1CondDBIOVWriter);
DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyDummyProd);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1TriggerKeyListDummyProd);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProd);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(L1TriggerConfigOnlineProd);

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondTools/L1Trigger/interface/WriterProxy.h"

DEFINE_SEAL_MODULE();

// Central L1 records
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

REGISTER_PLUGIN(L1TriggerKeyRcd, L1TriggerKey);
REGISTER_L1_WRITER(L1TriggerKeyRcd, L1TriggerKey);

#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"

REGISTER_PLUGIN(L1TriggerKeyListRcd, L1TriggerKeyList);
REGISTER_L1_WRITER(L1TriggerKeyListRcd, L1TriggerKeyList);

// L1 scales
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"

REGISTER_PLUGIN(L1JetEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1JetEtScaleRcd, L1CaloEtScale);

REGISTER_PLUGIN(L1EmEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1EmEtScaleRcd, L1CaloEtScale);

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"

REGISTER_PLUGIN(L1MuTriggerScalesRcd, L1MuTriggerScales);
REGISTER_L1_WRITER(L1MuTriggerScalesRcd, L1MuTriggerScales);

#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

REGISTER_PLUGIN(L1MuTriggerPtScaleRcd, L1MuTriggerPtScale);
REGISTER_L1_WRITER(L1MuTriggerPtScaleRcd, L1MuTriggerPtScale);

#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"

REGISTER_PLUGIN(L1MuGMTScalesRcd, L1MuGMTScales);
REGISTER_L1_WRITER(L1MuGMTScalesRcd, L1MuGMTScales);

// DT TPG records
#include "CondFormats/L1TObjects/interface/DTConfigManager.h"
#include "CondFormats/DataRecord/interface/DTConfigManagerRcd.h"

REGISTER_PLUGIN(DTConfigManagerRcd, DTConfigManager);
REGISTER_L1_WRITER(DTConfigManagerRcd, DTConfigManager);

// DT TF records
#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPtaLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTEtaPatternLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTExtLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTPhiLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTPtaLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTQualPatternLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"

REGISTER_PLUGIN(L1MuDTEtaPatternLutRcd, L1MuDTEtaPatternLut);
REGISTER_L1_WRITER(L1MuDTEtaPatternLutRcd, L1MuDTEtaPatternLut);

REGISTER_PLUGIN(L1MuDTExtLutRcd, L1MuDTExtLut);
REGISTER_L1_WRITER(L1MuDTExtLutRcd, L1MuDTExtLut);

REGISTER_PLUGIN(L1MuDTPhiLutRcd, L1MuDTPhiLut);
REGISTER_L1_WRITER(L1MuDTPhiLutRcd, L1MuDTPhiLut);

REGISTER_PLUGIN(L1MuDTPtaLutRcd, L1MuDTPtaLut);
REGISTER_L1_WRITER(L1MuDTPtaLutRcd, L1MuDTPtaLut);

REGISTER_PLUGIN(L1MuDTQualPatternLutRcd, L1MuDTQualPatternLut);
REGISTER_L1_WRITER(L1MuDTQualPatternLutRcd, L1MuDTQualPatternLut);

REGISTER_PLUGIN(L1MuDTTFParametersRcd, L1MuDTTFParameters);
REGISTER_L1_WRITER(L1MuDTTFParametersRcd, L1MuDTTFParameters);

// CSC TPG records
#include "CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"

REGISTER_PLUGIN(L1CSCTPParametersRcd, L1CSCTPParameters);
REGISTER_L1_WRITER(L1CSCTPParametersRcd, L1CSCTPParameters);

// CSC TF records
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCDTLut.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCGlobalLuts.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCLocalPhiLut.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCDTLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCGlobalLutsRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCLocalPhiLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"

REGISTER_PLUGIN(L1MuCSCTFConfigurationRcd, L1MuCSCTFConfiguration);
REGISTER_L1_WRITER(L1MuCSCTFConfigurationRcd, L1MuCSCTFConfiguration);

REGISTER_PLUGIN(L1MuCSCDTLutRcd, L1MuCSCDTLut);
REGISTER_L1_WRITER(L1MuCSCDTLutRcd, L1MuCSCDTLut);

REGISTER_PLUGIN(L1MuCSCGlobalLutsRcd, L1MuCSCGlobalLuts);
REGISTER_L1_WRITER(L1MuCSCGlobalLutsRcd, L1MuCSCGlobalLuts);

REGISTER_PLUGIN(L1MuCSCLocalPhiLutRcd, L1MuCSCLocalPhiLut);
REGISTER_L1_WRITER(L1MuCSCLocalPhiLutRcd, L1MuCSCLocalPhiLut);

REGISTER_PLUGIN(L1MuCSCPtLutRcd, L1MuCSCPtLut);
REGISTER_L1_WRITER(L1MuCSCPtLutRcd, L1MuCSCPtLut);

// RPC records
// #include "CondFormats/RPCObjects/interface/L1RPCConfig.h"
// #include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"

// REGISTER_PLUGIN(L1RPCConfigRcd, L1RPCConfig);
// REGISTER_L1_WRITER(L1RPCConfigRcd, L1RPCConfig);

// GMT records
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
#include "CondFormats/DataRecord/interface/L1MuGMTParametersRcd.h"

REGISTER_PLUGIN(L1MuGMTParametersRcd, L1MuGMTParameters);
REGISTER_L1_WRITER(L1MuGMTParametersRcd, L1MuGMTParameters);

// RCT records
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

REGISTER_PLUGIN(L1RCTParametersRcd, L1RCTParameters);
REGISTER_L1_WRITER(L1RCTParametersRcd, L1RCTParameters);

#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

REGISTER_PLUGIN(L1RCTChannelMaskRcd, L1RCTChannelMask);
REGISTER_L1_WRITER(L1RCTChannelMaskRcd, L1RCTChannelMask);

#include "CondFormats/L1TObjects/interface/L1RctInputScale.h"
#include "CondFormats/DataRecord/interface/L1RctEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1RctHcalScaleRcd.h"

REGISTER_PLUGIN(L1RctEcalScaleRcd, L1RctInputScale);
REGISTER_L1_WRITER(L1RctEcalScaleRcd, L1RctInputScale);

REGISTER_PLUGIN(L1RctHcalScaleRcd, L1RctInputScale);
REGISTER_L1_WRITER(L1RctHcalScaleRcd, L1RctInputScale);

// GCT records
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

REGISTER_PLUGIN(L1GctJetFinderParamsRcd, L1GctJetFinderParams);
REGISTER_L1_WRITER(L1GctJetFinderParamsRcd, L1GctJetFinderParams);

#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"

REGISTER_PLUGIN(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);
REGISTER_L1_WRITER(L1GctJetCalibFunRcd, L1GctJetEtCalibrationFunction);

#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterNegativeEtaRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCounterPositiveEtaRcd.h"

REGISTER_PLUGIN(L1GctJetCounterNegativeEtaRcd, L1GctJetCounterSetup);
REGISTER_L1_WRITER(L1GctJetCounterNegativeEtaRcd, L1GctJetCounterSetup);

REGISTER_PLUGIN(L1GctJetCounterPositiveEtaRcd, L1GctJetCounterSetup);
REGISTER_L1_WRITER(L1GctJetCounterPositiveEtaRcd, L1GctJetCounterSetup);



// GT records
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

REGISTER_PLUGIN(L1GtBoardMapsRcd, L1GtBoardMaps);
REGISTER_L1_WRITER(L1GtBoardMapsRcd, L1GtBoardMaps);

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

REGISTER_PLUGIN(L1GtParametersRcd, L1GtParameters);
REGISTER_L1_WRITER(L1GtParametersRcd, L1GtParameters);

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

REGISTER_PLUGIN(L1GtPrescaleFactorsAlgoTrigRcd, L1GtPrescaleFactors);
REGISTER_L1_WRITER(L1GtPrescaleFactorsAlgoTrigRcd, L1GtPrescaleFactors);
REGISTER_PLUGIN(L1GtPrescaleFactorsTechTrigRcd, L1GtPrescaleFactors);
REGISTER_L1_WRITER(L1GtPrescaleFactorsTechTrigRcd, L1GtPrescaleFactors);

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

REGISTER_PLUGIN(L1GtStableParametersRcd, L1GtStableParameters);
REGISTER_L1_WRITER(L1GtStableParametersRcd, L1GtStableParameters);

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

REGISTER_PLUGIN(L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask);
REGISTER_L1_WRITER(L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask);
REGISTER_PLUGIN(L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask);
REGISTER_L1_WRITER(L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask);

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

REGISTER_PLUGIN(L1GtTriggerMenuRcd, L1GtTriggerMenu);
REGISTER_L1_WRITER(L1GtTriggerMenuRcd, L1GtTriggerMenu);

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

REGISTER_PLUGIN(L1CaloGeometryRecord, L1CaloGeometry);
REGISTER_L1_WRITER(L1CaloGeometryRecord, L1CaloGeometry);
