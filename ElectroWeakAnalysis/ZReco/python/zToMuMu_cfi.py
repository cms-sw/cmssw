import FWCore.ParameterSet.Config as cms

zToMuMu = cms.EDFilter("MassMinCandShallowCloneCombiner",
    massMin = cms.double(20.0),
    decay = cms.string('allMuons@+ allMuons@-')
)


