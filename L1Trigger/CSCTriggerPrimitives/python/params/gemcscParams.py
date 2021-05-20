import FWCore.ParameterSet.Config as cms

# GEM coincidence pad processors
copadParamGE11 = cms.PSet(
    verbosity = cms.uint32(0),
    maxDeltaPad = cms.uint32(4),
    maxDeltaRoll = cms.uint32(1),
    maxDeltaBX = cms.uint32(0)
)

copadParamGE21 = copadParamGE11.clone()

## LUTs for the Run-3 GEM-CSC integrated local trigger
gemcscParams = cms.PSet(

    ## convert pad number to 1/2-strip in ME1a
    padToHsME1aFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_hs_ME1a_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_hs_ME1a_odd.txt",
    ),
    ## convert pad number to 1/2-strip in ME1b
    padToHsME1bFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_hs_ME1b_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_hs_ME1b_odd.txt",
    ),
    ## convert pad number to 1/2-strip in ME21
    padToHsME21Files = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_hs_ME21_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_hs_ME21_odd.txt",
    ),
    ## convert pad number to 1/8-strip in ME1a
    padToEsME1aFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1a_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1a_odd.txt",
    ),
    ## convert pad number to 1/8-strip in ME1b
    padToEsME1bFiles = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1b_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1b_odd.txt",
    ),
    ## convert pad number to 1/8-strip in ME21
    padToEsME21Files = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME21_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME21_odd.txt",
    ),
    ## convert eta partition to minimum wiregroup in ME11
    rollToMinWgME11Files = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME11_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME11_odd.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_min_wg_ME11_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_min_wg_ME11_odd.txt",
    ),
    ## convert eta partition to maximum wiregroup in ME11
    rollToMaxWgME11Files = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME11_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME11_odd.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_max_wg_ME11_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_max_wg_ME11_odd.txt",
    ),
     ## convert eta partition to minimum wiregroup in ME21
    rollToMinWgME21Files = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME21_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME21_odd.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_min_wg_ME21_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_min_wg_ME21_odd.txt",
    ),
    ## convert eta partition to maximum wiregroup in ME21
    rollToMaxWgME21Files = cms.vstring(
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME21_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME21_odd.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_max_wg_ME21_even.txt",
        "L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_max_wg_ME21_odd.txt",
    ),
)

gemcscPSets = cms.PSet(
    copadParamGE11 = copadParamGE11.clone(),
    copadParamGE21 = copadParamGE21.clone(),
    gemcscParams = gemcscParams.clone()
)
