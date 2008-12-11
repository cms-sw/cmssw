import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import *
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

from RecoEgamma.EgammaIsolationAlgos.eleEcalExtractorBlocks_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleHcalExtractorBlocks_cff import *

eleIsoDepositTk.ExtractorPSet.ComponentName = cms.string('TrackExtractor')
eleIsoDepositEcalSCVetoFromClusts = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pixelMatchGsfElectrons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( EleIsoEcalSCVetoFromClustsExtractorBlock )
)
eleIsoDepositEcalFromClusts = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pixelMatchGsfElectrons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( EleIsoEcalFromClustsExtractorBlock )
)
eleIsoDepositHcalFromTowers = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pixelMatchGsfElectrons"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet( EleIsoHcalFromTowersExtractorBlock )
)

# define module labels for old (tk-based isodeposit) POG isolation
# WARNING: these labels are used in the functions below.
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

from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import eleIsoFromDepsEcalFromHits
def useElectronRecHitIsolation(process,layers=(0,1,)):
    if (layers.__contains__(0)):
        print "Switching to ECAL RecHit isolation for Electrons in PAT Layer 0"

        # Replace the existing input tag with the one from RecHits (keeping the same order)
        index = patAODElectronIsolationLabels.index(cms.InputTag("eleIsoDepositEcalFromClusts"))
        patAODElectronIsolationLabels.pop(index)
        patAODElectronIsolationLabels.insert(index,cms.InputTag("eleIsoDepositEcalFromHits"))

        # Assign the new labels
        process.patAODElectronIsolations.associations = patAODElectronIsolationLabels
        process.layer0ElectronIsolations.associations = patAODElectronIsolationLabels

        # Redefine the path: we don't need cluster mergers anymore
        process.patAODElectronIsolation = cms.Sequence( patAODElectronIsolations )

        # Get all this back in the layer-0 electrons
        process.allLayer0Electrons.isolation.ecal.src = cms.InputTag("patAODElectronIsolations","eleIsoDepositEcalFromHits")

    if (layers.__contains__(1)):
        print "Switching to ECAL RecHit isolation for Electrons in PAT Layer 1"
        process.allLayer1Electrons.isolation.ecal.src = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits")
        process.allLayer1Electrons.isoDeposits.ecal   = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits")
        # Use the recommended RecHit isolation cuts from E/gamma
        process.allLayer1Electrons.isolation.ecal.vetos  = eleIsoFromDepsEcalFromHits.deposits[0].vetos
        process.allLayer1Electrons.isolation.ecal.deltaR = eleIsoFromDepsEcalFromHits.deposits[0].deltaR


