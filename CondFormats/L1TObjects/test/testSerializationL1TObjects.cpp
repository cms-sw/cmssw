#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{

  testSerialization<L1TMuonOverlapParams>();  
  testSerialization<L1TMuonBarrelParams>();  
  testSerialization<L1TMuonGlobalParams>();  
  testSerialization<l1t::CaloParams>();
  testSerialization<l1t::CaloConfig>();
    testSerialization<L1CaloEcalScale>();
    testSerialization<L1CaloEtScale>();
    testSerialization<L1CaloGeometry>();
    testSerialization<L1CaloHcalScale>();
    testSerialization<L1GctChannelMask>();
    testSerialization<L1GctJetFinderParams>();
    testSerialization<L1GtAlgorithm>();
    testSerialization<L1GtBoard>();
    testSerialization<L1GtBoardMaps>();
    testSerialization<L1GtBptxTemplate>();
    testSerialization<L1GtCaloTemplate>();
    testSerialization<L1GtCaloTemplate::CorrelationParameter>();
    testSerialization<L1GtCaloTemplate::ObjectParameter>();
    testSerialization<L1GtCastorTemplate>();
    testSerialization<L1GtCondition>();
    testSerialization<L1GtCorrelationTemplate>();
    testSerialization<L1GtCorrelationTemplate::CorrelationParameter>();
    testSerialization<L1GtEnergySumTemplate>();
    testSerialization<L1GtEnergySumTemplate::ObjectParameter>();
    testSerialization<L1GtExternalTemplate>();
    testSerialization<L1GtHfBitCountsTemplate>();
    testSerialization<L1GtHfBitCountsTemplate::ObjectParameter>();
    testSerialization<L1GtHfRingEtSumsTemplate>();
    testSerialization<L1GtHfRingEtSumsTemplate::ObjectParameter>();
    testSerialization<L1GtJetCountsTemplate>();
    testSerialization<L1GtJetCountsTemplate::ObjectParameter>();
    testSerialization<L1GtMuonTemplate>();
    testSerialization<L1GtMuonTemplate::CorrelationParameter>();
    //testSerialization<L1GtMuonTemplate::ObjectParameter>(); has uninitialized booleans
    testSerialization<L1GtParameters>();
    testSerialization<L1GtPrescaleFactors>();
    testSerialization<L1GtPsbConfig>();
    testSerialization<L1GtPsbSetup>();
    testSerialization<L1GtStableParameters>();
    testSerialization<L1GtTriggerMask>();
    testSerialization<L1GtTriggerMenu>();
    testSerialization<L1MuBinnedScale>();
    testSerialization<L1MuCSCPtLut>();
    testSerialization<L1MuCSCTFAlignment>();
    testSerialization<L1MuCSCTFConfiguration>();
    testSerialization<L1MuDTEtaPattern>();
    testSerialization<L1MuDTEtaPatternLut>();
    testSerialization<L1MuDTExtLut>();
    testSerialization<L1MuDTExtLut::LUT>();
    testSerialization<L1MuDTPhiLut>();
    testSerialization<L1MuDTPtaLut>();
    testSerialization<L1MuDTQualPatternLut>();
    testSerialization<L1MuDTTFMasks>();
    testSerialization<L1MuDTTFParameters>();
    testSerialization<L1MuGMTChannelMask>();
    //testSerialization<L1MuGMTParameters>(); has uninitialized booleans
    testSerialization<L1MuGMTScales>();
    //testSerialization<L1MuPacking>(); abstract
    testSerialization<L1MuPseudoSignedPacking>();
    //testSerialization<L1MuScale>(); abstract
    testSerialization<L1MuSymmetricBinnedScale>();
    testSerialization<L1MuTriggerPtScale>();
    testSerialization<L1MuTriggerScales>();
    //testSerialization<L1RCTChannelMask>(); has uninitialized booleans
    //testSerialization<L1RCTNoisyChannelMask>(); has uninitialized booleans
    //testSerialization<L1RCTParameters>(); has uninitialized booleans
    testSerialization<L1RPCBxOrConfig>();
    testSerialization<L1RPCConeDefinition>();
    testSerialization<L1RPCConeDefinition::TLPSize>();
    testSerialization<L1RPCConeDefinition::TRingToLP>();
    testSerialization<L1RPCConeDefinition::TRingToTower>();
    testSerialization<L1RPCConfig>();
    testSerialization<L1RPCHsbConfig>();
    testSerialization<L1TriggerKey>();
    testSerialization<L1TriggerKeyList>();
    testSerialization<RPCPattern>();
    testSerialization<RPCPattern::RPCLogicalStrip>();
    testSerialization<RPCPattern::TQuality>();
    testSerialization<std::map<int, std::vector<L1GtObject>>>();
    testSerialization<std::map<int, std::vector<L1GtObject>>::value_type>();
    testSerialization<std::map<short,L1MuDTEtaPattern>>();
    testSerialization<std::map<short,L1MuDTEtaPattern>::value_type>();
    testSerialization<std::map<std::string, L1GtAlgorithm>>();
    testSerialization<std::map<std::string, L1GtAlgorithm>::value_type>();
    testSerialization<std::pair<int, std::vector<L1GtObject>>>();
    testSerialization<std::pair<short,L1MuDTEtaPattern>>();
    testSerialization<std::pair<std::string, L1GtAlgorithm>>();
    testSerialization<std::vector<L1GtBoard>>();
    testSerialization<std::vector<L1GtBptxTemplate>>();
    testSerialization<std::vector<L1GtCaloTemplate::ObjectParameter>>();
    testSerialization<std::vector<L1GtCaloTemplate>>();
    testSerialization<std::vector<L1GtCastorTemplate>>();
    testSerialization<std::vector<L1GtCorrelationTemplate>>();
    testSerialization<std::vector<L1GtEnergySumTemplate::ObjectParameter>>();
    testSerialization<std::vector<L1GtEnergySumTemplate>>();
    testSerialization<std::vector<L1GtExternalTemplate>>();
    testSerialization<std::vector<L1GtHfBitCountsTemplate::ObjectParameter>>();
    testSerialization<std::vector<L1GtHfBitCountsTemplate>>();
    testSerialization<std::vector<L1GtHfRingEtSumsTemplate::ObjectParameter>>();
    testSerialization<std::vector<L1GtHfRingEtSumsTemplate>>();
    testSerialization<std::vector<L1GtJetCountsTemplate::ObjectParameter>>();
    testSerialization<std::vector<L1GtJetCountsTemplate>>();
    testSerialization<std::vector<L1GtMuonTemplate::ObjectParameter>>();
    testSerialization<std::vector<L1GtMuonTemplate>>();
    testSerialization<std::vector<L1GtPsbConfig>>();
    testSerialization<std::vector<L1GtPsbQuad>>();
    testSerialization<std::vector<L1MuDTExtLut::LUT>>();
    testSerialization<std::vector<L1RPCConeDefinition::TLPSize>>();
    testSerialization<std::vector<L1RPCConeDefinition::TRingToLP>>();
    testSerialization<std::vector<L1RPCConeDefinition::TRingToTower>>();
    testSerialization<std::vector<RPCPattern::TQuality>>();
    testSerialization<std::vector<RPCPattern>>();
    testSerialization<std::vector<std::vector<L1GtBptxTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtCaloTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtCastorTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtCorrelationTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtEnergySumTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtExternalTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtHfBitCountsTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtHfRingEtSumsTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtJetCountsTemplate>>>();
    testSerialization<std::vector<std::vector<L1GtMuonTemplate>>>();
    testSerialization<L1TUtmAlgorithm>();
    testSerialization<L1TUtmBin>();
    testSerialization<L1TUtmCondition>();
    testSerialization<L1TUtmCut>();
    testSerialization<L1TUtmCutValue>();
    testSerialization<L1TUtmObject>();
    testSerialization<L1TUtmScale>();
    testSerialization<L1TUtmTriggerMenu>();
    testSerialization<L1TGlobalPrescalesVetos>();
    testSerialization<L1TGlobalParameters>();


    return 0;
}
