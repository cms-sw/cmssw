import FWCore.ParameterSet.Config as cms

hltEgammaCombMassFilter = cms.EDFilter("HLTEgammaCombMassFilter",
                                            firstLegLastFilter = cms.InputTag("firstFilter"),
                                            secondLegLastFilter = cms.InputTag("secondFilter"),
                                            minMass = cms.double(-1.0)
)


