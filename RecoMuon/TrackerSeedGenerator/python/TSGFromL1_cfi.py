import FWCore.ParameterSet.Config as cms

hltL3TrajectorySeedFromL1 = cms.EDFilter("TSGFromL1Muon",
    FitterPSet = cms.PSet(

    ),
    OrderedHitsFactoryPSet = cms.PSet(

    ),
    L1MuonLabel = cms.InputTag("l1extraParticles")
)


