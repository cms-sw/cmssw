import FWCore.ParameterSet.Config as cms

from RecoTICL.SeedingRegions.ticlSeedingRegionProducer_cfi import ticlSeedingRegionProducer
from RecoTICL.LayerClustering.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoTICL.Configuration.MIPStep_cff import ticlTrackstersMIP, filteredLayerClustersMIP
from RecoTICL.LayerClustering.hgcalLayerClusters_cff import (hgcalSoARecHitsEE, hgcalCLUEsteringEE,
                                                          hgcalSoALayerClustersEE, hgcalMergeLayerClusters)


def customiseTICLForMuonCassettesTest(process):
    # The CE-E layer clusters only, with lower thresholds than the default ones: the rechit threshold (ecut, in units
    # of the noise), the density radius (deltac, in cm) and the seed density (kappa, in units of the noise)
    process.hgcalSoARecHitsMuonCassettesTestEE = hgcalSoARecHitsEE.clone(ecut = 1.5)
    process.hgcalCLUEsteringMuonCassettesTestEE = hgcalCLUEsteringEE.clone(
        hgcalRecHitsSoA = 'hgcalSoARecHitsMuonCassettesTestEE',
        deltac = 1.0,
        kappa = 5.
    )
    process.hgcalSoALayerClustersMuonCassettesTestEE = hgcalSoALayerClustersEE.clone(
        hgcalRecHitsSoA = 'hgcalSoARecHitsMuonCassettesTestEE',
        hgcalRecHitsLayerClustersSoA = 'hgcalCLUEsteringMuonCassettesTestEE'
    )

    process.hgcalMergeLayerClusters = hgcalMergeLayerClusters.clone(
        layerClustersSoA = ['hgcalSoALayerClustersMuonCassettesTestEE'],
        hgcalRecHitsSoA = ['hgcalSoARecHitsMuonCassettesTestEE'],
        hgcalRecHitsLayerClustersSoA = ['hgcalCLUEsteringMuonCassettesTestEE'],
        extraLayerClusters = [],
        extraTimeLayerClusters = []
    )

    process.hgcalLocalRecoTask = cms.Task(process.HGCalRecHit, process.HGCalUncalibRecHit,
                                          process.hgcalSoARecHitsMuonCassettesTestEE,
                                          process.hgcalCLUEsteringMuonCassettesTestEE,
                                          process.hgcalSoALayerClustersMuonCassettesTestEE,
                                          process.hgcalMergeLayerClusters, process.recHitMapProducer,
                                          process.ticlGeomESProducer, process.ticlGeomWithBarrelESProducer)

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
        energy_em_over_total_threshold = cms.float(-1),
        etaLimitIncreaseWindow = cms.float(2.1),
        filter_on_categories = cms.vint32(0),
        max_delta_time = cms.float(-1),
        max_longitudinal_sigmaPCA = cms.float(9999),
        max_missing_layers_in_trackster = cms.int32(9999),
        max_out_in_hops = cms.int32(10),
        min_cos_pointing = cms.float(0.5),
        min_cos_theta = cms.float(0.9),
        min_layers_per_trackster = cms.int32(4),
        oneTracksterPerTrackSeed = cms.bool(False),
        out_in_dfs = cms.bool(False),
        pid_threshold = cms.float(0),
        promoteEmptyRegionToTrackster = cms.bool(False),
        root_doublet_max_distance_from_seed_squared = cms.float(9999),
        shower_start_max_layer = cms.int32(9999),
        siblings_maxRSquared = cms.vfloat(0.0006, 0.0006, 0.0006),
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
    process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hgcalSoALayerClustersMuonCassettesTestEE_*_*')
    process.schedule = cms.Schedule(process.raw2digi_step,process.FEVTDEBUGHLToutput_step)
    process.schedule.associate(process.ticlMIPStepTask, process.hgcalLocalRecoTask, process.ticlLayerTileTask)
    return process
