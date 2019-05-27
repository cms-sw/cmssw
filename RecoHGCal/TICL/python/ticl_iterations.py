import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

# patch particle flow clusters for HGC into local reco sequence
# (for now until global reco is going with some sort of clustering)
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import hgcalMultiClusters


from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer
from RecoHGCal.TICL.multiClustersFromTrackstersProducer_cfi import multiClustersFromTrackstersProducer


def TICL_iterations_withReco(process):
  process.FEVTDEBUGHLTEventContent.outputCommands.extend(['keep *_MultiClustersFromTracksters*_*_*'])

  process.FilteredLayerClustersMIP = filteredLayerClustersProducer.clone(
      clusterFilter = "ClusterFilterBySize",
      algo_number = 8,
      max_cluster_size = 2, # inclusive
      iteration_label = "MIP"
  )

  process.TrackstersMIP = trackstersProducer.clone(
      filtered_mask = cms.InputTag("FilteredLayerClustersMIP", "MIP"),
      missing_layers = 3,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.99, # ~10 degrees
      min_cos_pointing = 0.9
  )

  process.MultiClustersFromTrackstersMIP = multiClustersFromTrackstersProducer.clone(
      label = "MIPMultiClustersFromTracksterByCA",
      Tracksters = "TrackstersMIP"
  )

  process.FilteredLayerClusters = filteredLayerClustersProducer.clone(
      algo_number = 8,
      iteration_label = "algo8",
      LayerClustersInputMask = "TrackstersMIP"
  )

  process.Tracksters = trackstersProducer.clone(
      original_mask = "TrackstersMIP",
      filtered_mask = cms.InputTag("FilteredLayerClusters", "algo8"),
      missing_layers = 2,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.94, # ~20 degrees
      min_cos_pointing = 0.7
  )

  process.MultiClustersFromTracksters = multiClustersFromTrackstersProducer.clone(
      Tracksters = "Tracksters"
  )

  process.hgcalMultiClusters = hgcalMultiClusters
  process.TICL_Task = cms.Task(
      process.FilteredLayerClustersMIP,
      process.TrackstersMIP,
      process.MultiClustersFromTrackstersMIP,
      process.FilteredLayerClusters,
      process.Tracksters,
      process.MultiClustersFromTracksters)
  process.schedule.associate(process.TICL_Task)
  return process

def TICL_iterations(process):
  process.FEVTDEBUGHLTEventContent.outputCommands.extend(['keep *_MultiClustersFromTracksters*_*_*'])

  process.FilteredLayerClustersMIP = filteredLayerClustersProducer.clone(
      clusterFilter = "ClusterFilterBySize",
      algo_number = 8,
      max_cluster_size = 2, # inclusive
      iteration_label = "MIP"
  )

  process.TrackstersMIP = trackstersProducer.clone(
      filtered_mask = cms.InputTag("FilteredLayerClustersMIP", "MIP"),
      missing_layers = 3,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.985 # ~10 degrees
  )

  process.MultiClustersFromTrackstersMIP = multiClustersFromTrackstersProducer.clone(
      label = "MIPMultiClustersFromTracksterByCA",
      Tracksters = "TrackstersMIP"
  )

  process.FilteredLayerClusters = filteredLayerClustersProducer.clone(
      algo_number = 8,
      iteration_label = "algo8"
  )

  process.Tracksters = trackstersProducer.clone(
      original_mask = "TrackstersMIP",
      filtered_mask = cms.InputTag("FilteredLayerClusters", "algo8"),
      missing_layers = 2,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.94, # ~20 degrees
      min_cos_pointing = 0.7
  )

  process.MultiClustersFromTracksters = multiClustersFromTrackstersProducer.clone(
      Tracksters = "Tracksters"
  )

  process.HGCalUncalibRecHit = HGCalUncalibRecHit
  process.HGCalRecHit = HGCalRecHit
  process.hgcalLayerClusters = hgcalLayerClusters
  process.hgcalMultiClusters = hgcalMultiClusters
  process.TICL_Task = cms.Task(process.HGCalUncalibRecHit,
      process.HGCalRecHit,
      process.hgcalLayerClusters,
      process.FilteredLayerClustersMIP,
      process.TrackstersMIP,
      process.MultiClustersFromTrackstersMIP,
      process.FilteredLayerClusters,
      process.Tracksters,
      process.MultiClustersFromTracksters,
      process.hgcalMultiClusters)
  process.schedule = cms.Schedule(process.raw2digi_step,process.FEVTDEBUGHLToutput_step)
  process.schedule.associate(process.TICL_Task)
  return process

