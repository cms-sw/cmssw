import FWCore.ParameterSet.Config as cms

slimmedLowPtElectrons = cms.EDProducer("PATElectronSlimmer",
   src = cms.InputTag("selectedPatLowPtElectrons"),                                  
   dropSuperCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropBasicClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropPFlowClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropPreshowerClusters = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropSeedCluster = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropRecHits = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropCorrections = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropIsolations = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropShapes = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropSaturation = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropExtrapolations  = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   dropClassifications  = cms.string("0"), # you can put a cut to slim selectively, e.g. pt < 10
   linkToPackedPFCandidates = cms.bool(False), # remove for the moment
   recoToPFMap = cms.InputTag("reducedEgamma","reducedGsfElectronPfCandMap"), # Not used, but the plugin asks for it anyhow
   packedPFCandidates = cms.InputTag("packedPFCandidates"), # Not used, but the plugin asks for it anyhow
   saveNonZSClusterShapes = cms.string("1"), # save additional user floats: (sigmaIetaIeta,sigmaIphiIphi,sigmaIetaIphi,r9,e1x5_over_e5x5)_NoZS 
   reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
   reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
   modifyElectrons = cms.bool(False),
   modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

