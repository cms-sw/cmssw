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
   dropSaturation = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropExtrapolations  = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   dropClassifications  = cms.string("pt < 5"), # you can put a cut to slim selectively, e.g. pt < 10
   linkToPackedPFCandidates = cms.bool(True),

   # Compute and store Mini-Isolation.
   # Implemention and a description of parameters can be found in:
   # PhysicsTools/PatUtils/src/PFIsolation.cc
   computeMiniIso = cms.bool(True),
    # veto on candidates in deadcone only in endcap
   miniIsoParamsE = cms.vdouble(0.05, 0.2, 10.0, 0.0, 0.015, 0.015, 0.08, 0.0, 0.0),
   miniIsoParamsB = cms.vdouble(0.05, 0.2, 10.0, 0.0, 0.000, 0.000, 0.00, 0.0, 0.0),

   recoToPFMap = cms.InputTag("reducedEgamma","reducedGsfElectronPfCandMap"),
   packedPFCandidates = cms.InputTag("packedPFCandidates"), 
   saveNonZSClusterShapes = cms.string("pt > 5"), # save additional user floats: (sigmaIetaIeta,sigmaIphiIphi,sigmaIetaIphi,r9,e1x5_over_e5x5)_NoZS 
   reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
   reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
   modifyElectrons = cms.bool(True),
   modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

