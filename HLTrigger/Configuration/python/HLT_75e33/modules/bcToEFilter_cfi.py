import FWCore.ParameterSet.Config as cms

bcToEFilter = cms.EDFilter("BCToEFilter",
    filterAlgoPSet = cms.PSet(
        eTThreshold = cms.double(10),
        genParSource = cms.InputTag("genParticles")
    )
)
