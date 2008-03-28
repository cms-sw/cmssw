import FWCore.ParameterSet.Config as cms

gtPsbTextToDigi = cms.EDFilter("GtPsbTextToDigi",
    FileEventOffset = cms.untracked.int32(0),
    TextFileName = cms.string('psb-')
)


