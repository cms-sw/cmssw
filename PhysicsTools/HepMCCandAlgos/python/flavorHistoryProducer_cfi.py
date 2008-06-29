import FWCore.ParameterSet.Config as cms

flavorHistoryProducer = cms.EDProducer("FlavorHistoryProducer",
                                       src = cms.InputTag("genParticles"),
                                       pdgIdToSelect = cms.int32(5),
                                       ptMinParticle = cms.double(5.0),
                                       ptMinShower = cms.double(5.0),
                                       etaMaxParticle = cms.double(4.0),
                                       etaMaxShower = cms.double(4.0),
                                       flavorHistoryName = cms.string("flavorHistory"),
                                       verbose = cms.untracked.bool(False)
                                       )
