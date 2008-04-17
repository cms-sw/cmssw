import FWCore.ParameterSet.Config as cms

# keys are used only in L1TriggerKeyDummyProd
# tags are used in L1CondDBIOVWriterProd and must match PoolDBESSource
L1SubsystemParams = cms.PSet(
    recordInfo = cms.VPSet(cms.PSet(
        record = cms.string('L1JetEtScaleRcd'),
        tag = cms.string('L1CaloEtScaleStandard'),
        type = cms.string('L1CaloEtScale'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1EmEtScaleRcd'),
        tag = cms.string('L1CaloEtScaleStandard'),
        type = cms.string('L1CaloEtScale'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuTriggerScalesRcd'),
        tag = cms.string('L1MuTriggerScalesStandard'),
        type = cms.string('L1MuTriggerScales'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuTriggerPtScaleRcd'),
        tag = cms.string('L1MuTriggerPtScaleStandard'),
        type = cms.string('L1MuTriggerPtScale'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuGMTScalesRcd'),
        tag = cms.string('L1MuGMTScalesStandard'),
        type = cms.string('L1MuGMTScales'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1CSCTPParametersRcd'),
        tag = cms.string('L1CSCTPParametersStandard'),
        type = cms.string('L1CSCTPParameters'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuDTEtaPatternLutRcd'),
        tag = cms.string('L1MuDTEtaPatternLutStandard'),
        type = cms.string('L1MuDTEtaPatternLut'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuDTExtLutRcd'),
        tag = cms.string('L1MuDTExtLutStandard'),
        type = cms.string('L1MuDTExtLut'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuDTPhiLutRcd'),
        tag = cms.string('L1MuDTPhiLutStandard'),
        type = cms.string('L1MuDTPhiLut'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuDTPtaLutRcd'),
        tag = cms.string('L1MuDTPtaLutStandard'),
        type = cms.string('L1MuDTPtaLut'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuDTQualPatternLutRcd'),
        tag = cms.string('L1MuDTQualPatternLutStandard'),
        type = cms.string('L1MuDTQualPatternLut'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1MuGMTParametersRcd'),
        tag = cms.string('L1MuGMTParametersStandard'),
        type = cms.string('L1MuGMTParameters'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1RCTParametersRcd'),
        tag = cms.string('L1RCTParametersStandard'),
        type = cms.string('L1RCTParameters'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GctJetFinderParamsRcd'),
        tag = cms.string('L1GctJetFinderParamsStandard'),
        type = cms.string('L1GctJetFinderParams'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GctJetCalibFunRcd'),
        tag = cms.string('L1GctJetEtCalibrationFunctionStandard'),
        type = cms.string('L1GctJetEtCalibrationFunction'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GctJetCounterNegativeEtaRcd'),
        tag = cms.string('L1GctJetCounterSetupStandard'),
        type = cms.string('L1GctJetCounterSetup'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GctJetCounterPositiveEtaRcd'),
        tag = cms.string('L1GctJetCounterSetupStandard'),
        type = cms.string('L1GctJetCounterSetup'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GtBoardMapsRcd'),
        tag = cms.string('L1GtBoardMapsStandard'),
        type = cms.string('L1GtBoardMaps'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GtParametersRcd'),
        tag = cms.string('L1GtParametersStandard'),
        type = cms.string('L1GtParameters'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GtPrescaleFactorsRcd'),
        tag = cms.string('L1GtPrescaleFactorsStandard'),
        type = cms.string('L1GtPrescaleFactors'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GtStableParametersRcd'),
        tag = cms.string('L1GtStableParametersStandard'),
        type = cms.string('L1GtStableParameters'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1GtTriggerMaskRcd'),
        tag = cms.string('L1GtTriggerMaskStandard'),
        type = cms.string('L1GtTriggerMask'),
        key = cms.string('dummy')
    ), cms.PSet(
        record = cms.string('L1CaloGeometryRecord'),
        tag = cms.string('L1CaloGeometryStandard'),
        type = cms.string('L1CaloGeometry'),
        key = cms.string('dummy')
    ))
)

