import FWCore.ParameterSet.Config as cms

goodZToMuMu = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string('goodMuons@+ goodMuons@-')
)


