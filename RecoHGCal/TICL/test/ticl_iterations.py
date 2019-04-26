import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

# patch particle flow clusters for HGC into local reco sequence
# (for now until global reco is going with some sort of clustering)
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import hgcalMultiClusters


from RecoHGCal.TICL.Tracksters_cfi import tracksters
from RecoHGCal.TICL.FilteredLayerClusters_cfi import FilteredLayerClusters
from RecoHGCal.TICL.MultiClustersFromTracksters_cfi import multiClustersFromTracksters


def TICL_iterations_withReco(process):
  process.FEVTDEBUGHLTEventContent.outputCommands.extend(['keep *_MultiClustersFromTracksters*_*_*'])

  process.FilteredLayerClustersMIP = FilteredLayerClusters.clone(
      ClusterFilter = "ClusterFilterBySize",
      algo_number = 8,
      max_cluster_size = 2, # inclusive
      iteration_label = "MIP"
  )

  process.TrackstersMIP = tracksters.clone(
      original_layerclusters_mask = cms.InputTag("hgcalLayerClusters", "InitialLayerClustersMask"),
      filtered_layerclusters_mask = cms.InputTag("FilteredLayerClustersMIP", "MIP"),
      algo_verbosity = 0,
      missing_layers = 3,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.99, # ~10 degrees
      min_cos_pointing = 0.9
  )

  process.MultiClustersFromTrackstersMIP = multiClustersFromTracksters.clone(
      label = "MIPMultiClustersFromTracksterByCA",
      Tracksters = cms.InputTag("TrackstersMIP", "TrackstersByCA")
  )

  process.FilteredLayerClusters = FilteredLayerClusters.clone(
      algo_number = 8,
      iteration_label = "algo8",
      LayerClustersInputMask = cms.InputTag("TrackstersMIP")
  )

  process.Tracksters = tracksters.clone(
      original_layerclusters_mask = cms.InputTag("TrackstersMIP"),
      filtered_layerclusters_mask = cms.InputTag("FilteredLayerClusters", "algo8"),
      algo_verbosity = 0,
      missing_layers = 2,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.94, # ~20 degrees
      min_cos_pointing = 0.7
  )

  process.MultiClustersFromTracksters = multiClustersFromTracksters.clone(
      Tracksters = cms.InputTag("Tracksters", "TrackstersByCA")
  )

  process.hgcalMultiClusters = hgcalMultiClusters
  process.TICL_Task = cms.Task(
      process.FilteredLayerClustersMIP,
      process.TrackstersMIP,
      process.MultiClustersFromTrackstersMIP,
      process.FilteredLayerClusters,
      process.Tracksters,
      process.MultiClustersFromTracksters)
  process.TICL = cms.Path(process.TICL_Task)
  return process

def TICL_iterations(process):
  process.FEVTDEBUGHLTEventContent.outputCommands.extend(['keep *_MultiClustersFromTracksters*_*_*'])

  process.FilteredLayerClustersMIP = FilteredLayerClusters.clone(
      ClusterFilter = "ClusterFilterBySize",
      algo_number = 8,
      max_cluster_size = 2, # inclusive
      iteration_label = "MIP"
  )

  process.TrackstersMIP = tracksters.clone(
      original_layerclusters_mask = cms.InputTag("hgcalLayerClusters", "InitialLayerClustersMask"),
      filtered_layerclusters_mask = cms.InputTag("FilteredLayerClustersMIP", "MIP"),
      algo_verbosity = 0,
      missing_layers = 3,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.985 # ~10 degrees
  )

  process.MultiClustersFromTrackstersMIP = multiClustersFromTracksters.clone(
      label = "MIPMultiClustersFromTracksterByCA",
      Tracksters = cms.InputTag("TrackstersMIP", "TrackstersByCA"),
  )

  process.FilteredLayerClusters = FilteredLayerClusters.clone(
      algo_number = 8,
      iteration_label = "algo8"
  )

  process.Tracksters = tracksters.clone(
      original_layerclusters_mask = cms.InputTag("TrackstersMIP"),
      filtered_layerclusters_mask = cms.InputTag("FilteredLayerClusters", "algo8"),
      algo_verbosity = 0,
      missing_layers = 2,
      min_clusters_per_ntuplet = 15,
      min_cos_theta = 0.94, # ~20 degrees
      min_cos_pointing = 0.7
  )

  process.MultiClustersFromTracksters = multiClustersFromTracksters.clone(
      Tracksters = cms.InputTag("Tracksters", "TrackstersByCA")
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
  process.TICL = cms.Path(process.TICL_Task)
  return process

