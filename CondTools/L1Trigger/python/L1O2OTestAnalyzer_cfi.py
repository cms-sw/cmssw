import FWCore.ParameterSet.Config as cms

L1O2OTestAnalyzer = cms.EDAnalyzer("L1O2OTestAnalyzer",
                                   printL1TriggerKey = cms.bool(True),
                                   printL1TriggerKeyList = cms.bool(True),
                                   printPayloadTokens = cms.bool(True),
                                   printESRecords = cms.bool(True),
                                   recordsToPrint = cms.vstring(
    'L1MuDTTFMasksRcd',
    'L1MuGMTChannelMaskRcd',
    'L1RCTChannelMaskRcd',
    'L1GctChannelMaskRcd',
    'L1GtPrescaleFactorsAlgoTrigRcd',
    'L1GtPrescaleFactorsTechTrigRcd',
    'L1GtTriggerMaskAlgoTrigRcd',
    'L1GtTriggerMaskTechTrigRcd',
    'L1GtTriggerMaskVetoTechTrigRcd' ) # Run Settings records
                                  )
