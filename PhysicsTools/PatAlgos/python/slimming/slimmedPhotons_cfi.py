import FWCore.ParameterSet.Config as cms

slimmedPhotons = cms.EDProducer("PATPhotonSlimmer",
    src = cms.InputTag("selectedPatPhotons"),
    dropSuperCluster = cms.string("0"), # always keep SC?   # ! (r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"), # you can put a cut to slim selectively, e.g. pt < 10
    dropBasicClusters = cms.string("! (r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"), # you can put a cut to slim selectively, e.g. pt < 10
    dropPreshowerClusters = cms.string("! (r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"), # you can put a cut to slim selectively, e.g. pt < 10
    dropSeedCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
    dropRecHits = cms.string("! (r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"), # you can put a cut to slim selectively, e.g. pt < 10
    linkToPackedPFCandidates = cms.bool(True),
    recoToPFMap = cms.InputTag("particleBasedIsolation","gedPhotons"),
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
)
