import FWCore.ParameterSet.Config as cms

CSCL1TPLookupTableEP = cms.ESSource("CSCL1TPLookupTableEP",
    esDiffToSlopeME11aFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11a_even_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11a_odd_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11a_even_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11a_odd_GEMlayer2.txt'
    ),
    esDiffToSlopeME11bFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11b_even_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11b_odd_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11b_even_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME11b_odd_GEMlayer2.txt'
    ),
    esDiffToSlopeME21Files = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME21_even_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME21_odd_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME21_even_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/BendingAngle/SlopeAmendment_ME21_odd_GEMlayer2.txt'
    ),
    gemCscSlopeCorrectionFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11a_even_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11b_even_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME21_even_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11a_odd_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11b_odd_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME21_odd_GEMlayer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11a_even_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11b_even_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME21_even_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11a_odd_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME11b_odd_GEMlayer2.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/ExtrapolationBySlope_ME21_odd_GEMlayer2.txt'
    ),
    gemCscSlopeCosiCorrectionFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/GEMCSCconsistentSlopeCorr_ME11a_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/GEMCSCconsistentSlopeCorr_ME11b_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/GEMCSCconsistentSlopeCorr_ME21_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/GEMCSCconsistentSlopeCorr_ME11a_odd_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/GEMCSCconsistentSlopeCorr_ME11b_odd_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/GEMCSCconsistentSlopeCorr_ME21_odd_layer1.txt'
    ),
    gemCscSlopeCosiFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_2to1_SlopeShift_ME11a_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_2to1_SlopeShift_ME11a_odd_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_3to1_SlopeShift_ME11a_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_3to1_SlopeShift_ME11a_odd_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_2to1_SlopeShift_ME11b_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_2to1_SlopeShift_ME11b_odd_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_3to1_SlopeShift_ME11b_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_3to1_SlopeShift_ME11b_odd_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_2to1_SlopeShift_ME21_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_2to1_SlopeShift_ME21_odd_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_3to1_SlopeShift_ME21_even_layer1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/SlopeCorrection/FacingChambers/CSCconsistency_3to1_SlopeShift_ME21_odd_layer1.txt'
    ),
    padToEsME11aFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1a_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1a_odd.txt'
    ),
    padToEsME11bFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1b_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME1b_odd.txt'
    ),
    padToEsME21Files = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME21_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_pad_es_ME21_odd.txt'
    ),
    positionLUTFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodePosOffsetLUT_pat0_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodePosOffsetLUT_pat1_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodePosOffsetLUT_pat2_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodePosOffsetLUT_pat3_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodePosOffsetLUT_pat4_v1.txt'
    ),
    rollToMaxWgME11Files = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME11_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME11_odd.txt'
    ),
    rollToMaxWgME21Files = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME21_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_max_wg_ME21_odd.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_max_wg_ME21_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_max_wg_ME21_odd.txt'
    ),
    rollToMinWgME11Files = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME11_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME11_odd.txt'
    ),
    rollToMinWgME21Files = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME21_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l1_min_wg_ME21_odd.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_min_wg_ME21_even.txt',
        'L1Trigger/CSCTriggerPrimitives/data/GEMCSC/CoordinateConversion/GEMCSCLUT_roll_l2_min_wg_ME21_odd.txt'
    ),
    slopeLUTFiles = cms.vstring(
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodeSlopeLUT_pat0_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodeSlopeLUT_pat1_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodeSlopeLUT_pat2_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodeSlopeLUT_pat3_v1.txt',
        'L1Trigger/CSCTriggerPrimitives/data/CCLUT/CSCComparatorCodeSlopeLUT_pat4_v1.txt'
    )
)
