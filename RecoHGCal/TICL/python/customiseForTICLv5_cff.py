import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClustersEE, hgcalLayerClustersHSi, hgcalLayerClustersHSci
from RecoLocalCalo.HGCalRecProducers.hgcalMergeLayerClusters_cfi import hgcalMergeLayerClusters
from RecoTracker.IterativeTracking.iterativeTk_cff import trackdnn_source
from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer

from RecoHGCal.TICL.CLUE3DEM_cff import *
from RecoHGCal.TICL.CLUE3DHAD_cff import *
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseForTICLv5EventContent
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, ticlIterLabelsMerge
from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper
from RecoHGCal.TICL.mergedTrackstersProducer_cfi import mergedTrackstersProducer as _mergedTrackstersProducer
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationLinkingbyCLUE3D as _tracksterSimTracksterAssociationLinkingbyCLUE3D
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationPRbyCLUE3D  as _tracksterSimTracksterAssociationPRbyCLUE3D
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit
from RecoHGCal.TICL.SimTracksters_cff import ticlSimTracksters, ticlSimTrackstersTask

from RecoHGCal.TICL.FastJetStep_cff import ticlTrackstersFastJet
from RecoHGCal.TICL.EMStep_cff import ticlTrackstersEM, ticlTrackstersHFNoseEM
from RecoHGCal.TICL.TrkStep_cff import ticlTrackstersTrk, ticlTrackstersHFNoseTrk
from RecoHGCal.TICL.MIPStep_cff import ticlTrackstersMIP, ticlTrackstersHFNoseMIP
from RecoHGCal.TICL.HADStep_cff import ticlTrackstersHAD, ticlTrackstersHFNoseHAD
from RecoHGCal.TICL.CLUE3DEM_cff import ticlTrackstersCLUE3DEM
from RecoHGCal.TICL.CLUE3DHAD_cff import ticlTrackstersCLUE3DHAD
from RecoHGCal.TICL.CLUE3DHighStep_cff import ticlTrackstersCLUE3DHigh
from RecoHGCal.TICL.TrkEMStep_cff import ticlTrackstersTrkEM, filteredLayerClustersHFNoseTrkEM

from RecoHGCal.TICL.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer

def customiseForTICLv5(process, enableDumper = False):

    process.HGCalUncalibRecHit.computeLocalTime = cms.bool(True)
    process.ticlSimTracksters.computeLocalTime = cms.bool(True)

    process.ticlTrackstersFastJet.pluginPatternRecognitionByFastJet.computeLocalTime = cms.bool(True)

    process.ticlTrackstersEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersTrk.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseTrk.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersMIP.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseMIP.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersHAD.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseHAD.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlTrackstersCLUE3DHAD.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)
    process.ticlTrackstersCLUE3DEM.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)
    process.ticlTrackstersCLUE3DHigh.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)

    process.ticlTrackstersTrkEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)
    process.ticlTrackstersHFNoseTrkEM.pluginPatternRecognitionByCA.computeLocalTime = cms.bool(True)

    process.ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

    process.ticlIterationsTask = cms.Task(
        ticlCLUE3DEMStepTask,
        ticlCLUE3DHADStepTask,
    )

    process.mtdSoA = _mtdSoAProducer.clone()
    process.mtdSoATask = cms.Task(process.mtdSoA)

    process.ticlTracksterLinks = _tracksterLinksProducer.clone()
    process.ticlTracksterLinksTask = cms.Task(process.ticlTracksterLinks)

    process.ticlCandidate = _ticlCandidateProducer.clone()
    process.ticlCandidateTask = cms.Task(process.ticlCandidate)

    process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DEM")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3DEM = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DEM")
        )
    process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3DHAD = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD")
        )

    process.mergedTrackstersProducer = _mergedTrackstersProducer.clone()

    process.tracksterSimTracksterAssociationLinkingbyCLUE3D = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("mergedTrackstersProducer")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3D = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("mergedTrackstersProducer")
        )
    process.iterTICLTask = cms.Task(process.ticlLayerTileTask,
                                     process.mtdSoATask,
                                     process.ticlIterationsTask,
                                     process.ticlTracksterLinksTask,
                                     process.ticlCandidateTask)
    process.particleFlowClusterHGCal.initialClusteringStep.tracksterSrc = "ticlCandidate"
    process.globalrecoTask.remove(process.ticlTrackstersMerge)

    process.tracksterSimTracksterAssociationLinking.label_tst = cms.InputTag("ticlCandidate")
    process.tracksterSimTracksterAssociationPR.label_tst = cms.InputTag("ticlCandidate")

    process.tracksterSimTracksterAssociationLinkingPU.label_tst = cms.InputTag("ticlCandidate")
    process.tracksterSimTracksterAssociationPRPU.label_tst = cms.InputTag("ticlCandidate")
    process.mergeTICLTask = cms.Task()
    process.pfTICL = _pfTICLProducer.clone(
      ticlCandidateSrc = cms.InputTag('ticlCandidate'),
      isTICLv5 = cms.bool(True)
    )
    process.hgcalAssociators = cms.Task(process.mergedTrackstersProducer, process.lcAssocByEnergyScoreProducer, process.layerClusterCaloParticleAssociationProducer,
                            process.scAssocByEnergyScoreProducer, process.layerClusterSimClusterAssociationProducer,
                            process.lcSimTSAssocByEnergyScoreProducer, process.layerClusterSimTracksterAssociationProducer,
                            process.simTsAssocByEnergyScoreProducer,  process.simTracksterHitLCAssociatorByEnergyScoreProducer,
                            process.tracksterSimTracksterAssociationLinking, process.tracksterSimTracksterAssociationPR,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3D, process.tracksterSimTracksterAssociationPRbyCLUE3D,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM, process.tracksterSimTracksterAssociationPRbyCLUE3DEM,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD, process.tracksterSimTracksterAssociationPRbyCLUE3DHAD,
                            process.tracksterSimTracksterAssociationLinkingPU, process.tracksterSimTracksterAssociationPRPU
                            )


    labelTst = ["ticlTrackstersCLUE3DEM", "ticlTrackstersCLUE3DHAD", "ticlTracksterLinks"]
    labelTst.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])
    lcInputMask  = ["ticlTrackstersCLUE3DEM", "ticlTrackstersCLUE3DHAD", "ticlTracksterLinks"]
    lcInputMask.extend([cms.InputTag("ticlSimTracksters", "fromCPs"), cms.InputTag("ticlSimTracksters")])
    process.hgcalValidator = hgcalValidator.clone(
        label_tst = cms.VInputTag(labelTst),
        LayerClustersInputMask = cms.VInputTag(lcInputMask),
        ticlTrackstersMerge = cms.InputTag("ticlCandidate"),
    )

    process.hgcalValidatorSequence = cms.Sequence(process.hgcalValidator)
    process.hgcalValidation = cms.Sequence(process.hgcalSimHitValidationEE+process.hgcalSimHitValidationHEF+process.hgcalSimHitValidationHEB+process.hgcalDigiValidationEE+process.hgcalDigiValidationHEF+process.hgcalDigiValidationHEB+process.hgcalRecHitValidationEE+process.hgcalRecHitValidationHEF+process.hgcalRecHitValidationHEB+process.hgcalHitValidationSequence+process.hgcalValidatorSequence+process.hgcalTiclPFValidation+process.hgcalPFJetValidation)
    process.globalValidationHGCal = cms.Sequence(process.hgcalValidation)
    process.validation_step9 = cms.EndPath(process.globalValidationHGCal)
    if(enableDumper):
        process.ticlDumper = ticlDumper.clone(
            saveLCs=True,
            saveCLUE3DTracksters=True,
            saveTrackstersMerged=True,
            saveSimTrackstersSC=True,
            saveSimTrackstersCP=True,
            saveTICLCandidate=True,
            saveSimTICLCandidate=True,
            saveTracks=True,
            saveAssociations=True,
            trackstersclue3d = cms.InputTag('mergedTrackstersProducer'),
            ticlcandidates = cms.InputTag("ticlCandidate"),
            trackstersmerged = cms.InputTag("ticlCandidate"),
            trackstersInCand = cms.InputTag("ticlCandidate")
        )
        process.TFileService = cms.Service("TFileService",
                                           fileName=cms.string("histo.root")
                                           )
        process.FEVTDEBUGHLToutput_step = cms.EndPath(
            process.FEVTDEBUGHLToutput + process.ticlDumper)


    process = customiseForTICLv5EventContent(process)

    return process

def customiseTICLv5FromReco(process, enableDumper = False):
    # TensorFlow ESSource

    process.TFESSource = cms.Task(process.trackdnn_source)

    process.hgcalLayerClustersTask = cms.Task(process.hgcalLayerClustersEE,
                                              process.hgcalLayerClustersHSi,
                                              process.hgcalLayerClustersHSci,
                                              process.hgcalMergeLayerClusters)

    # Reconstruction

    process.ticlSimTracksters.computeLocalTime = cms.bool(True)

    process.ticlTrackstersCLUE3DHAD.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)
    process.ticlTrackstersCLUE3DEM.pluginPatternRecognitionByCLUE3D.computeLocalTime = cms.bool(True)

    process.ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

    process.ticlIterationsTask = cms.Task(
        ticlCLUE3DEMStepTask,
        ticlCLUE3DHADStepTask,
    )

    process.mtdSoA = _mtdSoAProducer.clone()
    process.mtdSoATask = cms.Task(process.mtdSoA)

    process.ticlTracksterLinks = _tracksterLinksProducer.clone()
    process.ticlTracksterLinksTask = cms.Task(process.ticlTracksterLinks)

    process.ticlCandidate = _ticlCandidateProducer.clone()
    process.ticlCandidateTask = cms.Task(process.ticlCandidate)

    process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DEM")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3DEM = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DEM")
        )
    process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3DHAD = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHAD")
        )

    process.mergedTrackstersProducer = _mergedTrackstersProducer.clone()

    process.tracksterSimTracksterAssociationLinkingbyCLUE3D = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("mergedTrackstersProducer")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3D = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("mergedTrackstersProducer")
        )

    process.iterTICLTask = cms.Path(process.hgcalLayerClustersTask,
                            process.TFESSource,
                            process.ticlLayerTileTask,
                            process.mtdSoATask,
                            process.ticlIterationsTask,
                            process.ticlTracksterLinksTask,
                            process.ticlCandidateTask)

    process.particleFlowClusterHGCal.initialClusteringStep.tracksterSrc = "ticlCandidate"
    process.globalrecoTask.remove(process.ticlTrackstersMerge)

    process.tracksterSimTracksterAssociationLinking.label_tst = cms.InputTag("ticlCandidate")
    process.tracksterSimTracksterAssociationPR.label_tst = cms.InputTag("ticlCandidate")

    process.tracksterSimTracksterAssociationLinkingPU.label_tst = cms.InputTag("ticlCandidate")
    process.tracksterSimTracksterAssociationPRPU.label_tst = cms.InputTag("ticlCandidate")
    process.mergeTICLTask = cms.Task()
    process.pfTICL = _pfTICLProducerV5.clone()
    process.hgcalAssociators = cms.Task(process.hgcalRecHitMapProducer, process.mergedTrackstersProducer, process.lcAssocByEnergyScoreProducer, process.layerClusterCaloParticleAssociationProducer,
                            process.scAssocByEnergyScoreProducer, process.layerClusterSimClusterAssociationProducer,
                            process.lcSimTSAssocByEnergyScoreProducer, process.layerClusterSimTracksterAssociationProducer,
                            process.simTsAssocByEnergyScoreProducer,  process.simTracksterHitLCAssociatorByEnergyScoreProducer,
                            process.tracksterSimTracksterAssociationLinking, process.tracksterSimTracksterAssociationPR,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3D, process.tracksterSimTracksterAssociationPRbyCLUE3D,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM, process.tracksterSimTracksterAssociationPRbyCLUE3DEM,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD, process.tracksterSimTracksterAssociationPRbyCLUE3DHAD,
                            process.tracksterSimTracksterAssociationLinkingPU, process.tracksterSimTracksterAssociationPRPU
                            )

    if(enableDumper):
        process.ticlDumper = ticlDumper.clone(
            saveLCs=True,
            saveCLUE3DTracksters=True,
            saveTrackstersMerged=True,
            saveSimTrackstersSC=True,
            saveSimTrackstersCP=True,
            saveTICLCandidate=True,
            saveSimTICLCandidate=True,
            saveTracks=True,
            saveAssociations=True,
            trackstersclue3d = cms.InputTag('mergedTrackstersProducer'),
            ticlcandidates = cms.InputTag("ticlCandidate"),
            trackstersmerged = cms.InputTag("ticlCandidate"),
            trackstersInCand = cms.InputTag("ticlCandidate")
        )
        process.TFileService = cms.Service("TFileService",
                                           fileName=cms.string("histo.root")
                                           )

    process.FEVTDEBUGHLToutput_step = cms.EndPath(process.ticlDumper)

    process.TICL_Validation = cms.Path(process.ticlSimTrackstersTask, process.hgcalAssociators)

# Schedule definition
    process.schedule = cms.Schedule(process.iterTICLTask,
                                    process.TICL_Validation,
                                    process.FEVTDEBUGHLToutput_step)
    process = customiseForTICLv5EventContent(process)

    return process
