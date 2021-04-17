import FWCore.ParameterSet.Config as cms

## LUTs for the Run-3 CSC trigger
cclutParams = cms.PSet(
    positionLUTFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePosOffsetLUT_pat0_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePosOffsetLUT_pat1_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePosOffsetLUT_pat2_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePosOffsetLUT_pat3_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePosOffsetLUT_pat4_v1.txt"
    ),
    slopeLUTFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodeSlopeLUT_pat0_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodeSlopeLUT_pat1_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodeSlopeLUT_pat2_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodeSlopeLUT_pat3_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodeSlopeLUT_pat4_v1.txt"
    ),
    patternConversionLUTFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePatternConversionLUT_pat0_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePatternConversionLUT_pat1_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePatternConversionLUT_pat2_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePatternConversionLUT_pat3_v1.txt",
        "L1Trigger/CSCTriggerPrimitives/data/CSCComparatorCodePatternConversionLUT_pat4_v1.txt"
    ),
)
