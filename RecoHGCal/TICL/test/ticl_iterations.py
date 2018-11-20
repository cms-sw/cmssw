import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

# patch particle flow clusters for HGC into local reco sequence
# (for now until global reco is going with some sort of clustering)
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import hgcalMultiClusters


from RecoHGCal.TICL.Tracksters_cfi import Tracksters
from RecoHGCal.TICL.FilteredLayerClusters_cfi import FilteredLayerClusters
from RecoHGCal.TICL.TrackstersToMultiCluster_cfi import TrackstersToMultiCluster


def TICL_iterations(process):
  process.FEVTDEBUGHLTEventContent.outputCommands.extend(['keep *_TrackstersToMultiCluster*_*_*'])

  process.FilteredLayerClusters = FilteredLayerClusters.clone()
  process.FilteredLayerClusters.algo_number = 8
  process.FilteredLayerClusters.iteration_label = "algo8"
  process.Tracksters = Tracksters.clone()
  process.Tracksters.filteredLayerClusters = cms.InputTag("FilteredLayerClusters", "algo8")
  process.Tracksters.algo_verbosity = 0
  process.Tracksters.missing_layers = 2
  process.Tracksters.min_clusters_per_ntuplet = 15
  process.Tracksters.min_cos_theta = 0.8
  process.Tracksters.min_cos_pointing = 0.7
  process.TrackstersToMultiCluster = TrackstersToMultiCluster.clone()
  process.TrackstersToMultiCluster.Tracksters = cms.InputTag("Tracksters", "TrackstersByCA")

  process.FilteredLayerClustersMIP = FilteredLayerClusters.clone()
  process.FilteredLayerClustersMIP.algo_number = 9
  process.FilteredLayerClustersMIP.iteration_label = "algo9"
  process.TrackstersMIP = Tracksters.clone()
  process.TrackstersMIP.filteredLayerClusters = cms.InputTag("FilteredLayerClustersMIP", "algo9")
  process.TrackstersMIP.algo_verbosity = 0
  process.TrackstersMIP.missing_layers = 3
  process.TrackstersMIP.min_clusters_per_ntuplet = 15
  process.TrackstersMIP.min_cos_theta = 0.915
  process.TrackstersToMultiClusterMIP = TrackstersToMultiCluster.clone()
  process.TrackstersToMultiClusterMIP.label = "MIPMultiClustersFromTracksterByCA"
  process.TrackstersToMultiClusterMIP.Tracksters = cms.InputTag("TrackstersMIP", "TrackstersByCA")

  process.HGCalUncalibRecHit = HGCalUncalibRecHit
  process.HGCalRecHit = HGCalRecHit
  process.hgcalLayerClusters = hgcalLayerClusters
  process.hgcalLayerClusters.ecut = 5
  process.hgcalLayerClusters.splitFullHaloClusters = True
  process.hgcalLayerClusters.promote_single_nodes = True
  process.hgcalLayerClusters.ecut_miplike = 15
  process.hgcalLayerClusters.apply_cutoff_distance = False
  process.hgcalLayerClusters.cutoff_distance = 15.
  process.hgcalLayerClusters.verbosity = 0
  process.hgcalMultiClusters = hgcalMultiClusters
  process.TICL = cms.Path(process.HGCalUncalibRecHit
      + process.HGCalRecHit
      + process.hgcalLayerClusters
      + process.FilteredLayerClustersMIP
      + process.TrackstersMIP
      + process.TrackstersToMultiClusterMIP
      + process.FilteredLayerClusters
      + process.Tracksters
      + process.TrackstersToMultiCluster
      + process.hgcalMultiClusters)
  return process

