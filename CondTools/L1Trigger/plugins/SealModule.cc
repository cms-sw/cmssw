#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/L1Trigger/plugins/L1CondDBPayloadWriter.h"
#include "CondTools/L1Trigger/plugins/L1CondDBIOVWriter.h"
#include "CondTools/L1Trigger/plugins/L1TriggerKeyDummyProd.h"
#include "CondTools/L1Trigger/plugins/L1TriggerKeyListDummyProd.h"
#include "CondTools/L1Trigger/plugins/L1SubsystemKeysOnlineProd.h"
#include "CondTools/L1Trigger/plugins/L1TriggerKeyOnlineProd.h"

using namespace l1t;

DEFINE_FWK_MODULE(L1CondDBPayloadWriter);
DEFINE_FWK_MODULE(L1CondDBIOVWriter);
DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyDummyProd);
DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyListDummyProd);
DEFINE_FWK_EVENTSETUP_MODULE(L1SubsystemKeysOnlineProd);
DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProd);

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondTools/L1Trigger/interface/WriterProxy.h"

// Central L1 records
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

REGISTER_L1_WRITER(L1TriggerKeyRcd, L1TriggerKey);

#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"

REGISTER_L1_WRITER(L1TriggerKeyListRcd, L1TriggerKeyList);

// L1 scales
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

REGISTER_L1_WRITER(L1JetEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1EmEtScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1HtMissScaleRcd, L1CaloEtScale);
REGISTER_L1_WRITER(L1HfRingEtScaleRcd, L1CaloEtScale);

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"

REGISTER_L1_WRITER(L1MuTriggerScalesRcd, L1MuTriggerScales);

#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

REGISTER_L1_WRITER(L1MuTriggerPtScaleRcd, L1MuTriggerPtScale);

#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"

REGISTER_L1_WRITER(L1MuGMTScalesRcd, L1MuGMTScales);

// DT TF records
#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPtaLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTEtaPatternLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTExtLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTPhiLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTPtaLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTQualPatternLutRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

REGISTER_L1_WRITER(L1MuDTEtaPatternLutRcd, L1MuDTEtaPatternLut);
REGISTER_L1_WRITER(L1MuDTExtLutRcd, L1MuDTExtLut);
REGISTER_L1_WRITER(L1MuDTPhiLutRcd, L1MuDTPhiLut);
REGISTER_L1_WRITER(L1MuDTPtaLutRcd, L1MuDTPtaLut);
REGISTER_L1_WRITER(L1MuDTQualPatternLutRcd, L1MuDTQualPatternLut);
REGISTER_L1_WRITER(L1MuDTTFParametersRcd, L1MuDTTFParameters);
REGISTER_L1_WRITER(L1MuDTTFMasksRcd, L1MuDTTFMasks);

// CSC TF records
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFAlignment.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"

REGISTER_L1_WRITER(L1MuCSCTFConfigurationRcd, L1MuCSCTFConfiguration);
REGISTER_L1_WRITER(L1MuCSCTFAlignmentRcd, L1MuCSCTFAlignment);
REGISTER_L1_WRITER(L1MuCSCPtLutRcd, L1MuCSCPtLut);

// RPC records
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"

REGISTER_L1_WRITER(L1RPCConfigRcd, L1RPCConfig);

#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"

REGISTER_L1_WRITER(L1RPCConeDefinitionRcd, L1RPCConeDefinition);

#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"

REGISTER_L1_WRITER(L1RPCBxOrConfigRcd, L1RPCBxOrConfig);

#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"

REGISTER_L1_WRITER(L1RPCHsbConfigRcd, L1RPCHsbConfig);

// GMT records
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
#include "CondFormats/DataRecord/interface/L1MuGMTParametersRcd.h"

REGISTER_L1_WRITER(L1MuGMTParametersRcd, L1MuGMTParameters);

#include "CondFormats/L1TObjects/interface/L1MuGMTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1MuGMTChannelMaskRcd.h"

REGISTER_L1_WRITER(L1MuGMTChannelMaskRcd, L1MuGMTChannelMask);

// RCT records
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"

REGISTER_L1_WRITER(L1RCTParametersRcd, L1RCTParameters);

#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"

REGISTER_L1_WRITER(L1RCTChannelMaskRcd, L1RCTChannelMask);

#include "CondFormats/L1TObjects/interface/L1RCTNoisyChannelMask.h"
#include "CondFormats/DataRecord/interface/L1RCTNoisyChannelMaskRcd.h"

REGISTER_L1_WRITER(L1RCTNoisyChannelMaskRcd, L1RCTNoisyChannelMask);

#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"

REGISTER_L1_WRITER(L1CaloEcalScaleRcd, L1CaloEcalScale);

#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

REGISTER_L1_WRITER(L1CaloHcalScaleRcd, L1CaloHcalScale);

// GCT records
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

REGISTER_L1_WRITER(L1GctChannelMaskRcd, L1GctChannelMask);

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

REGISTER_L1_WRITER(L1GctJetFinderParamsRcd, L1GctJetFinderParams);

// GT records
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

REGISTER_L1_WRITER(L1GtBoardMapsRcd, L1GtBoardMaps);

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

REGISTER_L1_WRITER(L1GtParametersRcd, L1GtParameters);

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

REGISTER_L1_WRITER(L1GtPrescaleFactorsAlgoTrigRcd, L1GtPrescaleFactors);
REGISTER_L1_WRITER(L1GtPrescaleFactorsTechTrigRcd, L1GtPrescaleFactors);

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"

REGISTER_L1_WRITER(L1GtStableParametersRcd, L1GtStableParameters);

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

REGISTER_L1_WRITER(L1GtTriggerMaskAlgoTrigRcd, L1GtTriggerMask);
REGISTER_L1_WRITER(L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask);
REGISTER_L1_WRITER(L1GtTriggerMaskVetoAlgoTrigRcd, L1GtTriggerMask);
REGISTER_L1_WRITER(L1GtTriggerMaskVetoTechTrigRcd, L1GtTriggerMask);

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

REGISTER_L1_WRITER(L1GtTriggerMenuRcd, L1GtTriggerMenu);

#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"
#include "CondFormats/DataRecord/interface/L1GtPsbSetupRcd.h"

REGISTER_L1_WRITER(L1GtPsbSetupRcd, L1GtPsbSetup);

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

REGISTER_L1_WRITER(L1CaloGeometryRecord, L1CaloGeometry);
