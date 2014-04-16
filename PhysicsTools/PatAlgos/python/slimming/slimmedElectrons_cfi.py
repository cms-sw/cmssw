import FWCore.ParameterSet.Config as cms

slimmedElectrons = cms.EDProducer("PATElectronSlimmer",
   src = cms.InputTag("selectedPatElectrons"),
   dropSuperCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropBasicClusters = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropPFlowClusters = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropPreshowerClusters = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropSeedCluster = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropRecHits = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropCorrections = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropIsolations = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropShapes = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropExtrapolations  = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropClassifications  = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   linkToPackedPFCandidates = cms.bool(True),
   recoToPFMap = cms.InputTag("particleBasedIsolation","gedGsfElectrons"),
   packedPFCandidates = cms.InputTag("packedPFCandidates"), 
)

