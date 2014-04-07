import FWCore.ParameterSet.Config as cms

slimmedPhotons = cms.EDProducer("PATPhotonSlimmer",
    src = cms.InputTag("selectedPatPhotons"),
    dropSuperCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
    dropBasicClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
    dropPreshowerClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
    dropSeedCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
    dropRecHits = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
    linkToPackedPFCandidates = cms.bool(True),
    recoToPFMap = cms.InputTag("particleBasedIsolation","gedPhotons"),
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
)
