import FWCore.ParameterSet.Config as cms

source = cms.Source("TBRUInputSource",
    TriggerFedId = cms.untracked.int32(1023),
    nFeds = cms.untracked.int32(-1),
    firstEvent = cms.untracked.uint32(1),
    quiet = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/cmt/onlinedev/data/examples/RU0030349_000.root')
)

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

