import FWCore.ParameterSet.Config as cms

l1tdttpgClient = cms.EDFilter("L1TDTTPGClient",
    prescaleLS = cms.untracked.int32(-1),
    output_dir = cms.untracked.string('L1T/L1TDTTPG'),
    prescaleEvt = cms.untracked.int32(500)
)


