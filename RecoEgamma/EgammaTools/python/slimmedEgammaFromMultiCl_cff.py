import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.cleanedEcalDrivenGsfElectronsFromMultiCl_cfi import cleanedEcalDrivenGsfElectronsFromMultiCl
from RecoEgamma.EgammaTools.hgcalElectronIDValueMap_cff import hgcalElectronIDValueMap
from PhysicsTools.PatAlgos.PATElectronProducer_cfi import PATElectronProducer
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import slimmedElectrons
from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

hgcElectronID = hgcalElectronIDValueMap.clone(
    electrons = "cleanedEcalDrivenGsfElectronsFromMultiCl",
)
patElectronsFromMultiCl = PATElectronProducer.clone(
    electronSource             = "cleanedEcalDrivenGsfElectronsFromMultiCl",
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
selectedPatElectronsFromMultiCl = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("patElectronsFromMultiCl"),
    cut = cms.string("!isEB && pt >= 10."),
)
slimmedElectronsFromMultiCl = slimmedElectrons.clone(
    src = "selectedPatElectronsFromMultiCl",
    linkToPackedPFCandidates = False,
    saveNonZSClusterShapes   = "0",
    modifyElectrons          = False,
)

slimmedElectronsFromMultiClTask = cms.Task(
    cleanedEcalDrivenGsfElectronsFromMultiCl,
    hgcElectronID,
    patElectronsFromMultiCl,
    selectedPatElectronsFromMultiCl,
    slimmedElectronsFromMultiCl
)


from RecoEgamma.EgammaTools.hgcalPhotonIDValueMap_cff import hgcalPhotonIDValueMap
from PhysicsTools.PatAlgos.PATPhotonProducer_cfi import PATPhotonProducer
from PhysicsTools.PatAlgos.slimming.slimmedPhotons_cfi import slimmedPhotons

hgcPhotonID = hgcalPhotonIDValueMap.clone()

patPhotonsFromMultiCl = PATPhotonProducer.clone(
    photonSource           = "photonsFromMultiCl",
    electronSource         = "ecalDrivenGsfElectronsFromMultiCl",
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
selectedPatPhotonsFromMultiCl = cms.EDFilter("PATPhotonSelector",
    src = cms.InputTag("patPhotonsFromMultiCl"),
    cut = cms.string("!isEB && pt >= 15."),
)
slimmedPhotonsFromMultiCl = slimmedPhotons.clone(
    src = "selectedPatPhotonsFromMultiCl",
    linkToPackedPFCandidates = False,
    saveNonZSClusterShapes   = "0",
    modifyPhotons            = False,
)

slimmedPhotonsFromMultiClTask = cms.Task(
    hgcPhotonID,
    patPhotonsFromMultiCl,
    selectedPatPhotonsFromMultiCl,
    slimmedPhotonsFromMultiCl
)

slimmedEgammaFromMultiClTask = cms.Task(
    hgcalRecHitMapProducer,
    slimmedElectronsFromMultiClTask,
    slimmedPhotonsFromMultiClTask
)
