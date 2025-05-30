import FWCore.ParameterSet.Config as cms


from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal

from RecoHGCal.TICL.ticlSeedingRegionProducer_cfi import ticlSeedingRegionProducer
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer
from RecoHGCal.TICL.MIPStep_cff import ticlTrackstersMIP, filteredLayerClustersMIP


from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClustersEE, hgcalMergeLayerClusters
from RecoTracker.IterativeTracking.iterativeTk_cff import trackdnn_source
from RecoLocalCalo.HGCalRecProducers.recHitMapProducer_cfi import recHitMapProducer
from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer

from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.tracksterSelectionTf_cfi import *

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer
from RecoHGCal.Configuration.RecoHGCal_EventContent_cff import customiseForTICLv5EventContent
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabels
from RecoHGCal.TICL.mergedTrackstersProducer_cfi import mergedTrackstersProducer as _mergedTrackstersProducer
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociation_cfi import allTrackstersToSimTrackstersAssociationsByLCs  as _allTrackstersToSimTrackstersAssociationsByLCs
from SimCalorimetry.HGCalAssociatorProducers.TSToSimTSAssociationByHits_cfi import allTrackstersToSimTrackstersAssociationsByHits  as _allTrackstersToSimTrackstersAssociationsByHits

from SimCalorimetry.HGCalAssociatorProducers.SimClusterToCaloParticleAssociation_cfi import SimClusterToCaloParticleAssociation
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator
from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit
from RecoHGCal.TICL.SimTracksters_cff import ticlSimTracksters, ticlSimTrackstersTask
from RecoHGCal.TICL.MIPStep_cff import ticlTrackstersMIP, filteredLayerClustersMIP


def customiseTICLForMuonCassettesTest(process):
    # TensorFlow ESSource

    process.hgcalLayerClustersMuonCassettesTestEE = hgcalLayerClustersEE.clone(
            calculatePositionInAlgo = cms.bool(True),
        detector = cms.string('EE'),
        mightGet = cms.optional.untracked.vstring,
        nHitsTime = cms.uint32(3),
        plugin = cms.PSet(
            dEdXweights = cms.vdouble(
                0.0, 9.205, 11.129999999999999, 11.129999999999999, 11.129999999999999,
                11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999,
                11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999, 11.129999999999999,
                11.129999999999999, 11.129999999999999, 11.129999999999999, 13.2, 13.2,
                13.2, 13.2, 13.2, 13.2, 13.2,
                13.2, 35.745000000000005, 59.665000000000006, 60.7, 60.7,
                60.7, 60.7, 60.7, 60.7, 60.7,
                60.7, 60.7, 71.89, 83.08, 83.255,
                83.52000000000001, 83.61, 83.61, 83.61, 83.61,
                83.61, 83.61, 83.61
            ),
            # deltac = cms.vdouble(1.3, 1.3, 1.3, 0.0315),
            deltac = cms.vdouble(1., 1.3, 1.3, 0.0315),
            deltasi_index_regemfac = cms.int32(3),
            dependSensor = cms.bool(True),
            ecut = cms.double(1.5),
            fcPerEle = cms.double(0.00016020506),
            fcPerMip = cms.vdouble(
                2.06, 3.43, 5.15, 2.06, 3.43,
                5.15
            ),
            kappa = cms.double(5),
            maxNumberOfThickIndices = cms.uint32(6),
            noiseMip = cms.PSet(
                refToPSet_ = cms.string('HGCAL_noise_heback')
            ),
            noises = cms.vdouble(
                2000.0, 2400.0, 2000.0, 2000.0, 2400.0,
                2000.0
            ),
            positionDeltaRho2 = cms.double(1.69),
            sciThicknessCorrection = cms.double(0.69),
            thicknessCorrection = cms.vdouble(
                0.75, 0.76, 0.75, 0.85, 0.85,
                0.84
            ),
            thresholdW0 = cms.vdouble(2.9, 2.9, 2.9),
            type = cms.string('SiCLUE'),
            use2x2 = cms.bool(True),
            verbosity = cms.untracked.uint32(3)
        ),
        recHits = cms.InputTag("HGCalRecHit","HGCEERecHits"),
        timeClname = cms.string('timeLayerCluster')
    )
    
    process.hgcalMergeLayerClusters = hgcalMergeLayerClusters.clone(
        layerClusters = cms.VInputTag("hgcalLayerClustersMuonCassettesTestEE"),
        timeClname = cms.string('timeLayerCluster'),
        time_layerclusters = cms.VInputTag("hgcalLayerClustersMuonCassettesTestEE:timeLayerCluster")
    )

    process.hgcalLocalRecoTask = cms.Task(process.HGCalRecHit, process.HGCalUncalibRecHit, process.hgcalLayerClustersMuonCassettesTestEE, process.hgcalMergeLayerClusters, process.recHitMapProducer)

    process.ticlLayerTileProducer = ticlLayerTileProducer.clone()

    process.ticlSeedingGlobal = ticlSeedingRegionProducer.clone()

    process.filteredLayerClustersMIP = filteredLayerClustersMIP.clone(
      clusterFilter = "ClusterFilterBySize",
      max_cluster_size = 2, # inclusive
      iteration_label = "MIP",
    )


    process.ticlLayerTileTask = cms.Task(process.ticlLayerTileProducer)


    process.ticlTrackstersMIP = ticlTrackstersMIP.clone(
      detector = cms.string('HGCAL'),
      filtered_mask = cms.InputTag("filteredLayerClustersMIP","MIP"),
      itername = cms.string('MIP'),
      layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
      layer_clusters_tiles = cms.InputTag("ticlLayerTileProducer"),
      original_mask = cms.InputTag("hgcalMergeLayerClusters","InitialLayerClustersMask"),
      patternRecognitionBy = cms.string('CA'),
      pluginPatternRecognitionByCA = cms.PSet(
        algo_verbosity = cms.int32(0),
        computeLocalTime = cms.bool(True),
        energy_em_over_total_threshold = cms.double(-1),
        etaLimitIncreaseWindow = cms.double(2.1),
        filter_on_categories = cms.vint32(0),
        max_delta_time = cms.double(-1),
        max_longitudinal_sigmaPCA = cms.double(9999),
        max_missing_layers_in_trackster = cms.int32(9999),
        max_out_in_hops = cms.int32(10),
        min_cos_pointing = cms.double(0.5),
        min_cos_theta = cms.double(0.9),
        min_layers_per_trackster = cms.int32(4),
        oneTracksterPerTrackSeed = cms.bool(False),
        out_in_dfs = cms.bool(False),
        pid_threshold = cms.double(0),
        promoteEmptyRegionToTrackster = cms.bool(False),
        root_doublet_max_distance_from_seed_squared = cms.double(9999),
        shower_start_max_layer = cms.int32(9999),
        siblings_maxRSquared = cms.vdouble(0.0006, 0.0006, 0.0006),
        skip_layers = cms.int32(3),
        type = cms.string('CA')
      )
    )
    process.ticlMIPStepTask = cms.Task(process.filteredLayerClustersMIP, process.ticlSeedingGlobal, process.ticlTrackstersMIP)

    process.FEVTDEBUGHLToutput.outputCommands.append('drop *_*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('drop *_ticlTracksters*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('drop *_hgcalLayerClusters*_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_ticlTrackstersMIP_*_*')
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hgcalMergeLayerClusters_*_*'),
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_HGCalRecHit_*_*',)
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hgcalLayerClustersMuonCassettesTestEE_*_*')
    process.schedule = cms.Schedule(process.raw2digi_step,process.FEVTDEBUGHLToutput_step)
    process.schedule.associate(process.ticlMIPStepTask, process.hgcalLocalRecoTask, process.ticlLayerTileTask)
    return process
