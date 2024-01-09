import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer

from RecoHGCal.TICL.CLUE3DEM_cff import *
from RecoHGCal.TICL.CLUE3DHAD_cff import *
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseForTICLv5EventContent
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels, ticlIterLabelsMerge
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationLinkingbyCLUE3D as _tracksterSimTracksterAssociationLinkingbyCLUE3D
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationPRbyCLUE3D  as _tracksterSimTracksterAssociationPRbyCLUE3D 
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidatorv5 


def customiseForTICLv5(process):

    process.ticlLayerTileTask = cms.Task(ticlLayerTileProducer)

    process.ticlIterationsTask = cms.Task(
        ticlCLUE3DEMStepTask,
        ticlCLUE3DHADStepTask,
    )

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

    
    process.iterTICLTask = cms.Task(process.ticlLayerTileTask,
                                     process.ticlIterationsTask,
                                     process.ticlTracksterLinksTask,
                                     process.ticlCandidateTask)
    process.particleFlowClusterHGCal.initialClusteringStep.tracksterSrc = "ticlCandidate"
    process.globalrecoTask.remove(process.ticlTrackstersMerge)

    process.tracksterSimTracksterAssociationLinking.label_tst = cms.InputTag("ticlCandidate")
    process.tracksterSimTracksterAssociationPR.label_tst = cms.InputTag("ticlCandidate")

    process.tracksterSimTracksterAssociationLinkingPU.label_tst = cms.InputTag("ticlTracksterLinks")
    process.tracksterSimTracksterAssociationPRPU.label_tst = cms.InputTag("ticlTracksterLinks")
    process.mergeTICLTask = cms.Task()
    process.pfTICL.ticlCandidateSrc = cms.InputTag("ticlCandidate") 
    process.hgcalAssociators = cms.Task(process.lcAssocByEnergyScoreProducer, process.layerClusterCaloParticleAssociationProducer,
                            process.scAssocByEnergyScoreProducer, process.layerClusterSimClusterAssociationProducer,
                            process.lcSimTSAssocByEnergyScoreProducer, process.layerClusterSimTracksterAssociationProducer,
                            process.simTsAssocByEnergyScoreProducer,  process.simTracksterHitLCAssociatorByEnergyScoreProducer,
                            process.tracksterSimTracksterAssociationLinking, process.tracksterSimTracksterAssociationPR,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM, process.tracksterSimTracksterAssociationPRbyCLUE3DEM,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD, process.tracksterSimTracksterAssociationPRbyCLUE3DHAD,
                            process.tracksterSimTracksterAssociationLinkingPU, process.tracksterSimTracksterAssociationPRPU
                            )

    process.hgcalValidatorv5 = hgcalValidatorv5.clone(
        ticlTrackstersMerge = cms.InputTag("ticlCandidate"),
        trackstersclue3d = cms.InputTag("ticlTracksterLinks")
    )
    process.hgcalValidatorSequence = cms.Sequence(process.hgcalValidatorv5)
    process.hgcalValidation = cms.Sequence(process.hgcalSimHitValidationEE+process.hgcalSimHitValidationHEF+process.hgcalSimHitValidationHEB+process.hgcalDigiValidationEE+process.hgcalDigiValidationHEF+process.hgcalDigiValidationHEB+process.hgcalRecHitValidationEE+process.hgcalRecHitValidationHEF+process.hgcalRecHitValidationHEB+process.hgcalHitValidationSequence+process.hgcalValidatorSequence+process.hgcalTiclPFValidation+process.hgcalPFJetValidation)
    process.globalValidationHGCal = cms.Sequence(process.hgcalValidation)
    process.validation_step9 = cms.EndPath(process.globalValidationHGCal)

    process = customiseForTICLv5EventContent(process)

    return process
