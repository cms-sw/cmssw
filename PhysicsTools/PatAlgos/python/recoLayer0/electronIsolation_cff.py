import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaSuperClusterMerger_cfi import *
from RecoEgamma.EgammaIsolationAlgos.egammaBasicClusterMerger_cfi import *

egammaSuperClusterMerger.src = cms.VInputTag(
    cms.InputTag('hybridSuperClusters'), 
   #cms.InputTag('correctedHybridSuperClusters'), 
    cms.InputTag('multi5x5SuperClusters', 'multi5x5EndcapSuperClusters'),
   #cms.InputTag('multi5x5SuperClustersWithPreshower'),
   #cms.InputTag('correctedMulti5x5SuperClustersWithPreshower')
)
egammaBasicClusterMerger.src = cms.VInputTag(
    cms.InputTag('hybridSuperClusters',  'hybridBarrelBasicClusters'), 
    cms.InputTag('multi5x5BasicClusters','multi5x5EndcapBasicClusters')
)

# define module labels for old (tk-based isodeposit) POG isolation
patAODElectronIsolationLabels = cms.VInputTag(
        cms.InputTag("eleIsoDepositTk"),
      #  cms.InputTag("eleIsoDepositEcalFromHits"),
      #  cms.InputTag("eleIsoDepositHcalFromHits"),
        cms.InputTag("eleIsoDepositEcalFromClusts"),       # try these two if you want to compute them from AOD
        cms.InputTag("eleIsoDepositHcalFromTowers"),       # instead of reading the values computed in RECO
      # cms.InputTag("eleIsoDepositEcalSCVetoFromClust"), # somebody might want this one for ECAL instead
)

# read and convert to ValueMap<IsoDeposit> keyed to Candidate
patAODElectronIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    collection   = cms.InputTag("pixelMatchGsfElectrons"),
    associations = patAODElectronIsolationLabels,
)

# re-key ValueMap<IsoDeposit> to Layer 0 output
layer0ElectronIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    collection   = cms.InputTag("allLayer0Electrons"),
    backrefs     = cms.InputTag("allLayer0Electrons"),
    commonLabel  = cms.InputTag("patAODElectronIsolations"),
    associations = patAODElectronIsolationLabels,
)

# selecting POG modules that can run on top of AOD
eleIsoDepositAOD = cms.Sequence( eleIsoDepositTk * eleIsoDepositEcalFromClusts * eleIsoDepositHcalFromTowers)

# sequence to run on AOD before PAT
patAODElectronIsolation = cms.Sequence(
        egammaSuperClusterMerger *  ## 
        egammaBasicClusterMerger *  ## 
        eleIsoDepositAOD *          ## Not needed any more, we use values from RECO
        patAODElectronIsolations)

# sequence to run after the PAT cleaners
patLayer0ElectronIsolation = cms.Sequence(layer0ElectronIsolations)

def useElectronAODIsolation(process,layers=(0,1,)):
    if (layers.__contains__(0)):
        print "Switching to isolation computed from AOD info for Electrons in PAT Layer 0"
        patAODElectronIsolationLabels = cms.VInputTag(
            cms.InputTag("eleIsoDepositTk"),
            cms.InputTag("eleIsoDepositEcalFromClusts"),
            cms.InputTag("eleIsoDepositHcalFromTowers"),
        )
        process.patAODElectronIsolations.associations = patAODElectronIsolationLabels
        process.layer0ElectronIsolations.associations = patAODElectronIsolationLabels
        #This does not work yet :-(
        # process.patAODElectronIsolation = cms.Sequence(
        #    egammaSuperClusterMerger * egammaBasicClusterMerger +  
        #    eleIsoDepositAOD +
        #    patAODElectronIsolations
        # )
        process.allLayer0Electrons.isolation.ecal.src = cms.InputTag("patAODElectronIsolations","eleIsoDepositEcalFromClusts")
        process.allLayer0Electrons.isolation.hcal.src = cms.InputTag("patAODElectronIsolations","eleIsoDepositHcalFromTowers")
    if (layers.__contains__(1)):
        print "Switching to isolation computed from AOD info for Electrons in PAT Layer 1"
        process.allLayer1Electrons.isolation.ecal.src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromClusts")
        process.allLayer1Electrons.isolation.hcal.src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositHcalFromTowers")
        process.allLayer1Electrons.isoDeposits.ecal   = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromClusts")
        process.allLayer1Electrons.isoDeposits.hcal   = cms.InputTag("layer0ElectronIsolations","eleIsoDepositHcalFromTowers")

def useElectronRecHitIsolation(process,layers=(0,1,)):
    if (layers.__contains__(0)):
        print "Switching to RecHit isolation for Electrons in PAT Layer 0"
        patAODElectronIsolationLabels = cms.VInputTag(
            cms.InputTag("eleIsoDepositTk"),
            cms.InputTag("eleIsoDepositEcalFromHits"),
            cms.InputTag("eleIsoDepositHcalFromHits"),
        )
        process.patAODElectronIsolations.associations = patAODElectronIsolationLabels
        process.layer0ElectronIsolations.associations = patAODElectronIsolationLabels
        #This does not work yet :-(
        # process.patAODElectronIsolation = cms.Sequence( patAODElectronIsolations )
        process.allLayer0Electrons.isolation.ecal.src = cms.InputTag("patAODElectronIsolations","eleIsoDepositEcalFromHits")
        process.allLayer0Electrons.isolation.hcal.src = cms.InputTag("patAODElectronIsolations","eleIsoDepositHcalFromHits")
    if (layers.__contains__(1)):
        print "Switching to RecHit isolation for Electrons in PAT Layer 1"
        process.allLayer1Electrons.isolation.ecal.src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits")
        process.allLayer1Electrons.isolation.hcal.src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositHcalFromHits")
        process.allLayer1Electrons.isoDeposits.ecal   = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits")
        process.allLayer1Electrons.isoDeposits.hcal   = cms.InputTag("layer0ElectronIsolations","eleIsoDepositHcalFromHits")

