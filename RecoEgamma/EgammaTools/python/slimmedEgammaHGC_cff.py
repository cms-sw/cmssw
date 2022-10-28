import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.cleanedEcalDrivenGsfElectronsHGC_cfi import cleanedEcalDrivenGsfElectronsHGC
from RecoEgamma.EgammaTools.hgcalElectronIDValueMap_cff import hgcalElectronIDValueMap
from PhysicsTools.PatAlgos.PATElectronProducer_cfi import PATElectronProducer
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import slimmedElectrons
from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

hgcElectronID = hgcalElectronIDValueMap.clone(
    electrons = "cleanedEcalDrivenGsfElectronsHGC",
)
patElectronsHGC = PATElectronProducer.clone(
    electronSource             = "cleanedEcalDrivenGsfElectronsHGC",
    beamLineSrc                = "offlineBeamSpot",
    pvSrc                      = "offlinePrimaryVertices",
    addElectronID              = False,
    addGenMatch                = False,
    addMVAVariables            = False,
    embedGenMatch              = False,
    embedGsfElectronCore       = True,
    embedGsfTrack              = True,
    embedSuperCluster          = True,
    embedPflowSuperCluster     = False,
    embedSeedCluster           = True,
    embedBasicClusters         = False,
    embedPreshowerClusters     = False,
    embedPflowBasicClusters    = False,
    embedPflowPreshowerClusters= False,
    embedPFCandidate           = False,
    embedTrack                 = True,
    embedRecHits               = False,
    embedHighLevelSelection    = True,
    userData = cms.PSet(
        userClasses = cms.PSet( src = cms.VInputTag('')),
        userFloats  = cms.PSet( src = cms.VInputTag(
            [cms.InputTag("hgcElectronID", key) for key in hgcElectronID.variables]
        )),
        userInts  = cms.PSet( src = cms.VInputTag('')),
        userCands = cms.PSet( src = cms.VInputTag('')),
        userFunctions = cms.vstring(),
        userFunctionLabels = cms.vstring()
    ),
)
selectedPatElectronsHGC = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("patElectronsHGC"),
    cut = cms.string("!isEB && pt >= 10."),
)
slimmedElectronsHGC = slimmedElectrons.clone(
    src = "selectedPatElectronsHGC",
    linkToPackedPFCandidates = False,
    saveNonZSClusterShapes   = "0",
    modifyElectrons          = False,
)

slimmedElectronsHGCTask = cms.Task(
    cleanedEcalDrivenGsfElectronsHGC,
    hgcElectronID,
    patElectronsHGC,
    selectedPatElectronsHGC,
    slimmedElectronsHGC
)


from RecoEgamma.EgammaTools.hgcalPhotonIDValueMap_cff import hgcalPhotonIDValueMap
from PhysicsTools.PatAlgos.PATPhotonProducer_cfi import PATPhotonProducer
from PhysicsTools.PatAlgos.slimming.slimmedPhotons_cfi import slimmedPhotons

hgcPhotonID = hgcalPhotonIDValueMap.clone()

patPhotonsHGC = PATPhotonProducer.clone(
    photonSource           = "photonsHGC",
    electronSource         = "ecalDrivenGsfElectronsHGC",
    beamLineSrc            = "offlineBeamSpot",
    addPhotonID            = False,
    addGenMatch            = False,
    embedSuperCluster      = True,
    embedSeedCluster       = True,
    embedBasicClusters     = False,
    embedPreshowerClusters = False,
    embedRecHits           = False,
    saveRegressionData     = False,
    embedGenMatch          = False,
    isolationValues = cms.PSet(),
    userData = cms.PSet(
        userClasses = cms.PSet( src = cms.VInputTag('')),
        userFloats  = cms.PSet( src = cms.VInputTag(
            [cms.InputTag("hgcPhotonID", key) for key in hgcPhotonID.variables]
        )),
        userInts  = cms.PSet( src = cms.VInputTag('')),
        userCands = cms.PSet( src = cms.VInputTag('')),
        userFunctions = cms.vstring(),
        userFunctionLabels = cms.vstring()
    ),
)
selectedPatPhotonsHGC = cms.EDFilter("PATPhotonSelector",
    src = cms.InputTag("patPhotonsHGC"),
    cut = cms.string("!isEB && pt >= 15."),
)
slimmedPhotonsHGC = slimmedPhotons.clone(
    src = "selectedPatPhotonsHGC",
    linkToPackedPFCandidates = False,
    saveNonZSClusterShapes   = "0",
    modifyPhotons            = False,
)

slimmedPhotonsHGCTask = cms.Task(
    hgcPhotonID,
    patPhotonsHGC,
    selectedPatPhotonsHGC,
    slimmedPhotonsHGC
)

slimmedEgammaHGCTask = cms.Task(
    hgcalRecHitMapProducer,
    slimmedElectronsHGCTask,
    slimmedPhotonsHGCTask
)
