import FWCore.ParameterSet.Config as cms

source = cms.Source("TBRUInputSource",
    nFeds = cms.untracked.int32(-1),
    quiet = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/cmt/onlinedev/data/examples/RU0030349_000.root'),
    maxEvents = cms.untracked.int32(-1),
    firstEvent = cms.untracked.uint32(1),
    TriggerFedId = cms.untracked.int32(1023)
)


