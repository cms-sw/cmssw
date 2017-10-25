import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask, addKeepStatement

def addHGCalEgammaCollections(process):
    task = getPatAlgosToolsTask(process)

    addToProcessAndTask('patElectronsFromMultiCl',
        process.patElectrons.clone(
            electronSource = cms.InputTag("ecalDrivenGsfElectronsFromMultiCl"),
            addElectronID = cms.bool(False),
            addGenMatch      = cms.bool(False),
            embedGenMatch    = cms.bool(False),
            embedBasicClusters = cms.bool(False),
            embedPflowBasicClusters = cms.bool(False),
            embedRecHits = cms.bool(False),
            embedGsfTrack = cms.bool(True),
            isolationValues = cms.PSet(),
            isolationValuesNoPFId = cms.PSet(),
            addMVAVariables = cms.bool(False),
        ),
        process, task
    )
    from RecoEgamma.EgammaTools.HGCalElectronIDValueMap_cfi import HGCalElectronIDValueMap
    addToProcessAndTask('hgcElectronID', HGCalElectronIDValueMap, process, task)
    process.patElectronsFromMultiCl.userData.userFloats.src.extend([
        cms.InputTag("hgcElectronID", key) for key in HGCalElectronIDValueMap.variables
    ])
    addToProcessAndTask('selectedPatElectronsFromMultiCl',
        process.selectedPatElectrons.clone(
            src = cms.InputTag("patElectronsFromMultiCl"),
            cut = cms.string("!isEB && pt > 15"),
        ),
        process, task
    )
    addToProcessAndTask('slimmedElectronsFromMultiCl',
        process.slimmedElectrons.clone(
            src = cms.InputTag("selectedPatElectronsFromMultiCl"),
            linkToPackedPFCandidates = cms.bool(False),
            saveNonZSClusterShapes = cms.string("0"),
        ),
        process, task
    )
    addKeepStatement(process,
                     "keep *_slimmedElectrons_*_*",
                     ["keep *_slimmedElectronsFromMultiCl_*_*"]
                    )


    addToProcessAndTask('patPhotonsFromMultiCl',
        process.patPhotons.clone(
            photonSource = cms.InputTag("photonsFromMultiCl"),
            electronSource = cms.InputTag("ecalDrivenGsfElectronsFromMultiCl"),
            addPhotonID = cms.bool(False),
            addGenMatch = cms.bool(False),
            embedGenMatch = cms.bool(False),
            embedRecHits = cms.bool(False),
            embedBasicClusters = cms.bool(False),
            saveRegressionData = cms.bool(False),
            isolationValues = cms.PSet(),
        ),
        process, task
    )
    from RecoEgamma.EgammaTools.HGCalPhotonIDValueMap_cfi import HGCalPhotonIDValueMap
    addToProcessAndTask('hgcPhotonID', HGCalPhotonIDValueMap, process, task)
    process.patPhotonsFromMultiCl.userData.userFloats.src.extend([
        cms.InputTag("hgcPhotonID", key) for key in HGCalPhotonIDValueMap.variables
    ])
    addToProcessAndTask('selectedPatPhotonsFromMultiCl',
        process.selectedPatPhotons.clone(
            src = cms.InputTag("patPhotonsFromMultiCl"),
            cut = cms.string("!isEB && pt > 15"),
        ),
        process, task
    )
    process.selectedPatPhotonsFromMultiCl.cut = cms.string("!isEB()")
    addToProcessAndTask('slimmedPhotonsFromMultiCl',
        process.slimmedPhotons.clone(
            src = cms.InputTag("selectedPatPhotonsFromMultiCl"),
            linkToPackedPFCandidates = cms.bool(False),
        ),
        process, task
    )
    addKeepStatement(process,
                     "keep *_slimmedPhotons_*_*",
                     ["keep *_slimmedPhotonsFromMultiCl_*_*"]
                    )

    return process
