import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.cleanedEcalDrivenGsfElectronsFromMultiCl_cfi import cleanedEcalDrivenGsfElectronsFromMultiCl
from RecoEgamma.EgammaTools.hgcalElectronIDValueMap_cff import hgcalElectronIDValueMap
from PhysicsTools.PatAlgos.PATElectronProducer_cfi import PATElectronProducer
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import slimmedElectrons

hgcElectronID = hgcalElectronIDValueMap.clone(
    electrons = cms.InputTag("cleanedEcalDrivenGsfElectronsFromMultiCl"),
)
patElectronsFromMultiCl = PATElectronProducer.clone(
    electronSource = cms.InputTag("cleanedEcalDrivenGsfElectronsFromMultiCl"),
    beamLineSrc    = cms.InputTag("offlineBeamSpot"),
    pvSrc          = cms.InputTag("offlinePrimaryVertices"),
    addElectronID  = cms.bool(False),
    addGenMatch    = cms.bool(False),
    addMVAVariables= cms.bool(False),
    embedGenMatch              = cms.bool(False),
    embedGsfElectronCore       = cms.bool(True),
    embedGsfTrack              = cms.bool(True),
    embedSuperCluster          = cms.bool(True),
    embedPflowSuperCluster     = cms.bool(False),
    embedSeedCluster           = cms.bool(True),
    embedBasicClusters         = cms.bool(False),
    embedPreshowerClusters     = cms.bool(False),
    embedPflowBasicClusters    = cms.bool(False),
    embedPflowPreshowerClusters= cms.bool(False),
    embedPFCandidate           = cms.bool(False),
    embedTrack                 = cms.bool(True),
    embedRecHits               = cms.bool(False),
    embedHighLevelSelection    = cms.bool(True),
    userData = cms.PSet(
        userClasses = cms.PSet( src = cms.VInputTag('')),
        userFloats = cms.PSet( src = cms.VInputTag(
            [cms.InputTag("hgcElectronID", key) for key in hgcElectronID.variables]
        )),
        userInts = cms.PSet( src = cms.VInputTag('')),
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
    src = cms.InputTag("selectedPatElectronsFromMultiCl"),
    dropSuperCluster = cms.string("0"),
    dropBasicClusters = cms.string("0"),
    dropPFlowClusters = cms.string("0"),
    dropPreshowerClusters = cms.string("0"),
    dropSeedCluster = cms.string("0"),
    dropRecHits = cms.string("0"),
    dropCorrections = cms.string("pt < 5"),
    dropIsolations = cms.string("pt < 5"),
    dropShapes = cms.string("pt < 5"),
    dropSaturation = cms.string("pt < 5"),
    dropExtrapolations  = cms.string("pt < 5"),
    dropClassifications  = cms.string("pt < 5"),
    linkToPackedPFCandidates = cms.bool(False),
    recoToPFMap = cms.InputTag("reducedEgamma", "reducedGsfElectronPfCandMap"),
    packedPFCandidates = cms.InputTag("packedPFCandidates"), 
    saveNonZSClusterShapes = cms.string("0"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    modifyElectrons = cms.bool(False),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
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
    photonSource = cms.InputTag("photonsFromMultiCl"),
    electronSource = cms.InputTag("ecalDrivenGsfElectronsFromMultiCl"),
    beamLineSrc = cms.InputTag("offlineBeamSpot"),
    addPhotonID = cms.bool(False),
    addGenMatch = cms.bool(False),
    embedSuperCluster      = cms.bool(True),
    embedSeedCluster       = cms.bool(True),
    embedBasicClusters     = cms.bool(False),
    embedPreshowerClusters = cms.bool(False),
    embedRecHits           = cms.bool(False),
    saveRegressionData     = cms.bool(False),
    embedGenMatch          = cms.bool(False),
    isolationValues = cms.PSet(),
    userData = cms.PSet(
        userClasses = cms.PSet( src = cms.VInputTag('')),
        userFloats = cms.PSet( src = cms.VInputTag(
            [cms.InputTag("hgcPhotonID", key) for key in hgcPhotonID.variables]
        )),
        userInts = cms.PSet( src = cms.VInputTag('')),
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
    src = cms.InputTag("selectedPatPhotonsFromMultiCl"),
    dropSuperCluster = cms.string("0"),
    dropBasicClusters = cms.string("0"),
    dropPreshowerClusters = cms.string("0"),
    dropSeedCluster = cms.string("0"),
    dropRecHits = cms.string("0"),
    dropSaturation = cms.string("0"),
    dropRegressionData = cms.string("1"),
    linkToPackedPFCandidates = cms.bool(False),
    recoToPFMap = cms.InputTag("reducedEgamma","reducedPhotonPfCandMap"),
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
    saveNonZSClusterShapes = cms.string("0"),
    reducedBarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
    modifyPhotons = cms.bool(False),
    modifierConfig = cms.PSet( modifications = cms.VPSet() )
)

slimmedPhotonsFromMultiClTask = cms.Task(
    hgcPhotonID,
    patPhotonsFromMultiCl,
    selectedPatPhotonsFromMultiCl,
    slimmedPhotonsFromMultiCl
)

slimmedEgammaFromMultiClTask = cms.Task(
    slimmedElectronsFromMultiClTask,
    slimmedPhotonsFromMultiClTask
)
