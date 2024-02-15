import FWCore.ParameterSet.Config as cms

###OMTF emulator configuration with simple extrapolation algorithm
from L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi import simOmtfDigis

## add parameters to enable simple extrapolation algorithm
simOmtfDigis_extrapolSimple = simOmtfDigis.clone(
    noHitValueInPdf = cms.bool(True),
    minDtPhiQuality = cms.int32(2),
    minDtPhiBQuality = cms.int32(4),

    dtRefHitMinQuality = cms.int32(4),

    stubEtaEncoding = cms.string("bits"),

    usePhiBExtrapolationFromMB1 = cms.bool(True),
    usePhiBExtrapolationFromMB2 = cms.bool(True),
    useStubQualInExtr = cms.bool(False),
    useEndcapStubsRInExtr = cms.bool(False),
    useFloatingPointExtrapolation = cms.bool(False),

    extrapolFactorsFilename = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/ExtrapolationFactors_simple.xml"),
    sorterType = cms.string("byLLH"),
    ghostBusterType = cms.string("byRefLayer"), # byLLH byRefLayer GhostBusterPreferRefDt
    goldenPatternResultFinalizeFunction = cms.int32(10)
)
