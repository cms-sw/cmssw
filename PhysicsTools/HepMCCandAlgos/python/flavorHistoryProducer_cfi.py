import FWCore.ParameterSet.Config as cms

flavorHistoryProducer = cms.EDProducer("FlavorHistoryProducer",
                                       src = cms.InputTag("genParticles"),
                                       pdgIdToSelect = cms.int32(4),
                                       ptMinParticle = cms.double(0.0),
                                       ptMinShower = cms.double(0.0),
                                       etaMaxParticle = cms.double(4.0),
                                       etaMaxShower = cms.double(4.0),
                                       flavorHistoryName = cms.string("cpartonFlavorHistory"),
                                       verbose = cms.untracked.bool(True)
                                       )
