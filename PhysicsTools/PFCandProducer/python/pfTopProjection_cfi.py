import FWCore.ParameterSet.Config as cms

pfTopProjection = cms.EDProducer("PFTopProjectorPF2PAT",
                                 verbose = cms.untracked.bool( False ),
                                 name = cms.untracked.string("No Name"),
                                 topCollection = cms.InputTag(""),
                                 bottomCollection = cms.InputTag(""),
)
