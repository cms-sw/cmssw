import FWCore.ParameterSet.Config as cms

gtPsbTextToDigi = cms.EDProducer("GtPsbTextToDigi",
    FileEventOffset = cms.untracked.int32(0),
    TextFileName = cms.string('psb-')
)


