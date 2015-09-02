import FWCore.ParameterSet.Config as cms

eejFilter = cms.EDFilter("SingleLQGenFilter",
    src        = cms.untracked.InputTag("generator"),
    eej       = cms.bool(True),
    enuej     = cms.bool(False),
    nuenuej   = cms.bool(False),
    mumuj     = cms.bool(False),
    munumuj   = cms.bool(False),
    numunumuj = cms.bool(False)
)

enuejFilter = cms.EDFilter("SingleLQGenFilter",
    src        = cms.untracked.InputTag("generator"),
    eej       = cms.bool(False),
    enuej     = cms.bool(True),
    nuenuej   = cms.bool(False),
    mumuj     = cms.bool(False),
    munumuj   = cms.bool(False),
    numunumuj = cms.bool(False)
)

mumujFilter = cms.EDFilter("SingleLQGenFilter",
    src        = cms.untracked.InputTag("generator"),
    eej       = cms.bool(False),
    enuej     = cms.bool(False),
    nuenuej   = cms.bool(False),
    mumuj     = cms.bool(True),
    munumuj   = cms.bool(False),
    numunumuj = cms.bool(False)
)

munumujFilter = cms.EDFilter("SingleLQGenFilter",
    src        = cms.untracked.InputTag("generator"),
    eej       = cms.bool(False),
    enuej     = cms.bool(False),
    nuenuej   = cms.bool(False),
    mumuj     = cms.bool(False),
    munumuj   = cms.bool(True),
    numunumuj = cms.bool(False)
)
