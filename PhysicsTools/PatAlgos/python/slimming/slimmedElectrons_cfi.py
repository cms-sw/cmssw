import FWCore.ParameterSet.Config as cms

slimmedElectrons = cms.EDProducer("PATElectronSlimmer",
   src = cms.InputTag("selectedPatElectrons"),                                  
   dropSuperCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropBasicClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropPFlowClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropPreshowerClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropSeedCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropRecHits = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropCorrections = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropIsolations = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropShapes = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropExtrapolations  = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropClassifications  = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   linkToPackedPFCandidates = cms.bool(True),
   recoToPFMap = cms.InputTag("reducedEgamma","reducedGsfElectronPfCandMap"),
   packedPFCandidates = cms.InputTag("packedPFCandidates"), 
   saveNonZSClusterShapes = cms.string("pt > 5"), # save additional user floats: (sigmaIetaIeta,sigmaIphiIphi,sigmaIetaIphi,r9,e1x5_over_e5x5)_NoZS 
   reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
   reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
   modifyElectrons = cms.bool(True),
   modifierConfig = cms.PSet( modifications = cms.VPSet() ),
   puppiIsolationChargedHadrons = cms.InputTag("egmElectronPUPPIIsolation", "h+-DR030-BarVeto000-EndVeto001"),
   puppiIsolationNeutralHadrons = cms.InputTag("egmElectronPUPPIIsolation", "h0-DR030-BarVeto000-EndVeto000"),
   puppiIsolationPhotons = cms.InputTag("egmElectronPUPPIIsolation", "gamma-DR030-BarVeto000-EndVeto008"),
   puppiNoLeptonsIsolationChargedHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation", "h+-DR030-BarVeto000-EndVeto001"),
   puppiNoLeptonsIsolationNeutralHadrons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation", "h0-DR030-BarVeto000-EndVeto000"),
   puppiNoLeptonsIsolationPhotons = cms.InputTag("egmElectronPUPPINoLeptonsIsolation", "gamma-DR030-BarVeto000-EndVeto008")
)

