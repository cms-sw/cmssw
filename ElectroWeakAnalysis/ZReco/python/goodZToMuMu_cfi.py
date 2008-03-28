import FWCore.ParameterSet.Config as cms

goodZToMuMu = cms.EDFilter("CandViewShallowCloneCombiner",
    cut = cms.string('mass > 20'),
    decay = cms.string('goodMuons@+ goodMuons@-')
)


