import FWCore.ParameterSet.Config as cms

zToEE = cms.EDFilter("MassMinCandShallowCloneCombiner",
    massMin = cms.double(20.0),
    decay = cms.string('allElectrons@+ allElectrons@-')
)


