import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.ticlDumper_cfi import ticlDumper
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseForTICLv5EventContent
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationLinkingbyCLUE3D as _tracksterSimTracksterAssociationLinkingbyCLUE3D
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import tracksterSimTracksterAssociationPRbyCLUE3D  as _tracksterSimTracksterAssociationPRbyCLUE3D

def customiseTICLv5FromReco(process, enableDumper = False):
    # TensorFlow ESSource
    process.TFESSource = cms.Task(process.trackdnn_source)

    # Reconstruction
    process.hgcalLayerClustersTask = cms.Task(process.hgcalLayerClustersEE,
                                              process.hgcalLayerClustersHSi,
                                              process.hgcalLayerClustersHSci,
                                              process.hgcalMergeLayerClusters)

    process.ticlIterationsTask = cms.Task(
        process.ticlCLUE3DHighStepTask,
        process.ticlTracksterLinksTask,
        process.ticlPassthroughStepTask
    )

    process.mergeTICLTask = cms.Task()

    process.iterTICLTask = cms.Path(process.hgcalLayerClustersTask,
                            process.TFESSource,
                            process.ticlLayerTileTask,
                            process.mtdSoATask,
                            process.mergeTICLTask,
                            process.ticlIterationsTask,
                            process.ticlCandidateTask,
                            process.ticlPFTask)

    process.tracksterSimTracksterAssociationLinkingbyCLUE3DHigh = _tracksterSimTracksterAssociationLinkingbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHigh")
        )
    process.tracksterSimTracksterAssociationPRbyCLUE3DHigh = _tracksterSimTracksterAssociationPRbyCLUE3D.clone(
        label_tst = cms.InputTag("ticlTrackstersCLUE3DHigh")
        )

    '''for future CLUE3D separate iterations, merge collections and compute scores
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
    '''

    process.hgcalAssociators = cms.Task(process.recHitMapProducer, process.lcAssocByEnergyScoreProducer, process.layerClusterCaloParticleAssociationProducer,
                            process.scAssocByEnergyScoreProducer, process.layerClusterSimClusterAssociationProducer,
                            process.lcSimTSAssocByEnergyScoreProducer, process.layerClusterSimTracksterAssociationProducer,
                            process.simTsAssocByEnergyScoreProducer,  process.simTracksterHitLCAssociatorByEnergyScoreProducer,
                            process.tracksterSimTracksterAssociationLinking, process.tracksterSimTracksterAssociationPR,
                            process.tracksterSimTracksterAssociationLinkingbyCLUE3DHigh, process.tracksterSimTracksterAssociationPRbyCLUE3DHigh,
                            process.tracksterSimTracksterAssociationLinkingPU, process.tracksterSimTracksterAssociationPRPU
                            )

    '''for future CLUE3D separate iterations, merge collections and compute scores
    process.tracksterSimTracksterAssociationLinkingbyCLUE3D, process.tracksterSimTracksterAssociationPRbyCLUE3D,
    process.tracksterSimTracksterAssociationLinkingbyCLUE3DEM, process.tracksterSimTracksterAssociationPRbyCLUE3DEM,
    process.tracksterSimTracksterAssociationLinkingbyCLUE3DHAD, process.tracksterSimTracksterAssociationPRbyCLUE3DHAD,
    '''

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
            trackstersclue3d = cms.InputTag('ticlTrackstersCLUE3DHigh'),
            ticlcandidates = cms.InputTag("ticlCandidate"),
            trackstersmerged = cms.InputTag("ticlCandidate"),
            trackstersInCand = cms.InputTag("ticlCandidate")
        )
        process.TFileService = cms.Service("TFileService",
                                           fileName=cms.string("histo.root")
                                           )

        process.FEVTDEBUGHLToutput_step = cms.EndPath(process.ticlDumper)

    process.TICL_Validator = cms.Task(process.hgcalValidator)
    process.TICL_Validation = cms.Path(process.ticlSimTrackstersTask, process.hgcalAssociators, process.TICL_Validator)

    # Schedule definition
    process.schedule = cms.Schedule(process.iterTICLTask,
                                    process.TICL_Validation,
                                    process.FEVTDEBUGHLToutput_step)
    process = customiseForTICLv5EventContent(process)

    return process
