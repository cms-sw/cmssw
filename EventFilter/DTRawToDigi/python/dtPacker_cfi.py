import FWCore.ParameterSet.Config as cms

dtpacker = cms.EDFilter("DTDigiToRawModule",
    useStandardFEDid = cms.untracked.bool(True),
    debugMode = cms.untracked.bool(False),
    digibytype = cms.untracked.bool(True),
    minFEDid = cms.untracked.int32(770),
    maxFEDid = cms.untracked.int32(775)
)


