import FWCore.ParameterSet.Config as cms

DQMStore = cms.Service("DQMStore",
    verbose = cms.untracked.int32(0),
    # similar to LSBasedMode but for offline. Explicitly sets LumiFLag on all
    # MEs/modules that allow it (canSaveByLumi)
    saveByLumi = cms.untracked.bool(False),
    trackME = cms.untracked.string(""),
)
