import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.TICLSeedingRegions_cff import ticlSeedingGlobal, ticlSeedingGlobalHFNose
from RecoHGCal.TICL.trackstersProducer_cfi import trackstersProducer as _trackstersProducer
from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

# CLUSTER FILTERING/MASKING

filteredLayerClustersRecovery = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 2, # inclusive
    iteration_label = "Recovery",
    LayerClustersInputMask = 'ticlTrackstersCLUE3DHigh',
    algo_number = [6, 7, 8],
)

# PATTERN RECOGNITION

ticlTrackstersRecovery = _trackstersProducer.clone(
    filtered_mask = "filteredLayerClustersRecovery:Recovery",
    original_mask = 'ticlTrackstersCLUE3DHigh',
    seeding_regions = "ticlSeedingGlobal",
    itername = "Recovery",
    patternRecognitionBy = "Recovery",
    pluginPatternRecognitionByRecovery = dict (
        algo_verbosity = 0
    ),
    pluginInferenceAlgoTracksterInferenceByPFN = cms.PSet(
      algo_verbosity = cms.int32(0),
      onnxPIDModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/id_v0.onnx'),
      onnxEnergyModelPath = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/energy_v0.onnx'),
      inputNames = cms.vstring(
        'input',
        'input_tr_features'
      ),
      output_en = cms.vstring('enreg_output'),
      output_id = cms.vstring('pid_output'),
      eid_min_cluster_energy = cms.double(1),
      eid_n_layers = cms.int32(50),
      eid_n_clusters = cms.int32(10),
      doPID = cms.int32(0),
      doRegression = cms.int32(0),
      type = cms.string('TracksterInferenceByPFN')
    ),
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

ticlRecoveryStepTask = cms.Task(ticlSeedingGlobal
    ,filteredLayerClustersRecovery
    ,ticlTrackstersRecovery)
