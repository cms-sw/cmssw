import FWCore.ParameterSet.Config as cms

###OMTF emulator configuration with simple extrapolation algorithm
from L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi import simOmtfDigis

## add parameters to enable simple extrapolation algorithm
simOmtfDigis.noHitValueInPdf = cms.bool(True)
simOmtfDigis.minDtPhiQuality = cms.int32(2)
simOmtfDigis.minDtPhiBQuality = cms.int32(4)

simOmtfDigis.dtRefHitMinQuality = cms.int32(4)

simOmtfDigis.stubEtaEncoding = cms.string("bits")

simOmtfDigis.usePhiBExtrapolationFromMB1 = cms.bool(True)
simOmtfDigis.usePhiBExtrapolationFromMB2 = cms.bool(True)
simOmtfDigis.useStubQualInExtr = cms.bool(False)
simOmtfDigis.useEndcapStubsRInExtr = cms.bool(False)
simOmtfDigis.useFloatingPointExtrapolation = cms.bool(False)

simOmtfDigis.extrapolFactorsFilename = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/ExtrapolationFactors_simple.xml")
simOmtfDigis.sorterType = cms.string("byLLH")
simOmtfDigis.ghostBusterType = cms.string("byRefLayer") # byLLH byRefLayer GhostBusterPreferRefDt
simOmtfDigis.goldenPatternResultFinalizeFunction = cms.int32(10)
