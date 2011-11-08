import FWCore.ParameterSet.Config as cms

l1tdttpgClient = cms.EDAnalyzer("L1TDTTPGClient",
    input_dir = cms.untracked.string('L1T/L1TDTTPG'),
    prescaleLS = cms.untracked.int32(-1),
    output_dir = cms.untracked.string('L1T/L1TDTTPG/Tests'),
    prescaleEvt = cms.untracked.int32(500)
)


