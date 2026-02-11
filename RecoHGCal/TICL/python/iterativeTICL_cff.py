import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.FastJetStep_cff import *
from RecoHGCal.TICL.CLUE3DHighStep_cff import *
from RecoHGCal.TICL.MIPStep_cff import *
from RecoHGCal.TICL.TrkEMStep_cff import *
from RecoHGCal.TICL.TrkStep_cff import *
from RecoHGCal.TICL.EMStep_cff import *
from RecoHGCal.TICL.HADStep_cff import *
from RecoHGCal.TICL.CLUE3DEM_cff import *
from RecoHGCal.TICL.CLUE3DHAD_cff import *
from RecoHGCal.TICL.PRbyRecovery_cff import *

from RecoHGCal.TICL.ticlLayerTileProducer_cfi import ticlLayerTileProducer
from RecoHGCal.TICL.pfTICLProducer_cfi import pfTICLProducer as _pfTICLProducer

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.superclustering_cff import *
from RecoHGCal.TICL.ticlCandidateProducer_cfi import ticlCandidateProducer as _ticlCandidateProducer

from RecoHGCal.TICL.mtdSoAProducer_cfi import mtdSoAProducer as _mtdSoAProducer
from Configuration.ProcessModifiers.ticlv5_TrackLinkingGNN_cff import ticlv5_TrackLinkingGNN

from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

ticlLayerTileTask = cms.Task(ticlLayerTileProducer)


# TICLv5 is now the default configuration
ticlTracksterLinks = _tracksterLinksProducer.clone(
    tracksters_collections = cms.VInputTag(
        'ticlTrackstersCLUE3DHigh',
        'ticlTrackstersRecovery'
    ),
    linkingPSet = cms.PSet(
      cylinder_radius_sqr_split = cms.double(9),
      proj_distance_split = cms.double(5),
      track_time_quality_threshold = cms.double(0.5),
      min_num_lcs = cms.uint32(15),
      min_trackster_energy = cms.double(20),
      pca_quality_th = cms.double(0.85),
      dot_prod_th = cms.double(0.97),
      lower_boundary = cms.vdouble(20, 10),  
      upper_boundary = cms.vdouble(150, 100),  
      upper_distance_projective_sqr = cms.vdouble(4, 60),  
      lower_distance_projective_sqr = cms.vdouble(4, 60),  
      min_distance_z = cms.vdouble(35, 35),  
      upper_distance_projective_sqr_closest_points = cms.vdouble(5, 30),  
      lower_distance_projective_sqr_closest_points = cms.vdouble(10, 50),  
      max_z_distance_closest_points = cms.vdouble(35, 35),
      cylinder_radius_sqr = cms.vdouble(9, 15),  
      deltaRxy = cms.double(4.),
      algo_verbosity = cms.int32(0),
      type = cms.string('Skeletons')
    ),  
    regressionAndPid = cms.bool(False),
    inferenceAlgo = cms.string(''),
    pluginInferenceAlgoTracksterInferenceByDNN = cms.PSet(
        algo_verbosity = cms.int32(0),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        inputNames  = cms.vstring('input'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(1),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        onnxEnergyModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/linking/energy_v0.onnx'),
        onnxPIDModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/linking/id_v0.onnx'),
        type = cms.string('TracksterInferenceByDNN')
    ),
    pluginInferenceAlgoTracksterInferenceByPFN = cms.PSet(
        algo_verbosity = cms.int32(0),
        doPID = cms.int32(1),
        doRegression = cms.int32(1),
        inputNames  = cms.vstring('input','input_tr_features'),
        output_en   = cms.vstring('enreg_output'),
        output_id   = cms.vstring('pid_output'),
        eid_min_cluster_energy = cms.double(2.5),
        eid_n_clusters = cms.int32(10),
        eid_n_layers = cms.int32(50),
        onnxEnergyModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx'),
        onnxPIDModelPath = cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/id_v0.onnx'),
        type = cms.string('TracksterInferenceByPFN')
    )
)

ticlCandidate = _ticlCandidateProducer.clone(
    inferenceAlgo=cms.string('TracksterInferenceByPFN'),
    regressionAndPid = cms.bool(True),
    pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
        algo_verbosity=cms.int32(0),
        onnxPIDModelPath=cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/id_v0.onnx'),
        onnxEnergyModelPath=cms.string('RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx'),
        inputNames=cms.vstring('input', 'input_tr_features'),
        output_en=cms.vstring('enreg_output'),
        output_id=cms.vstring('pid_output'),
        eid_min_cluster_energy=cms.double(2.5),
        eid_n_layers=cms.int32(50),
        eid_n_clusters=cms.int32(10),
        doPID=cms.int32(1),
        doRegression=cms.int32(1),
        type=cms.string('TracksterInferenceByPFN')
    )
)

ticlv5_TrackLinkingGNN.toModify(ticlCandidate,
        interpretationDescPSet = cms.PSet(
            onnxTrkLinkingModelFirstDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/FirstDiskPropGNN_v0.onnx'),
            onnxTrkLinkingModelInterfaceDisk = cms.FileInPath('RecoHGCal/TICL/data/ticlv5/onnx_models/TrackLinking_GNN/InterfaceDiskPropGNN_v0.onnx'),
            inputNames = cms.vstring('x', 'edge_index', 'edge_attr'),
            output = cms.vstring('output'),
            delta_tk_ts = cms.double(0.1),
            thr_gnn = cms.double(0.5),
            type = cms.string('GNNLink')
        )
    )
mtdSoA = _mtdSoAProducer.clone()

# pfTICL uses ticlCandidate by default in v5
pfTICL = _pfTICLProducer.clone(
    ticlCandidateSrc = cms.InputTag('ticlCandidate'), 
    useTimingAverage=True
)



ticlPFTask = cms.Task(pfTICL)

# v5 iterations: CLUE3DHigh + Recovery
ticlIterationsTask = cms.Task(
    ticlCLUE3DHighStepTask,
    ticlRecoveryStepTask
)



# Default labels for v5
ticlIterLabels = ["ticlTrackstersCLUE3DHigh", "ticlTracksterLinks", "ticlCandidate"]
ticlTracksterLinksTask = cms.Task(ticlTracksterLinks, ticlSuperclusteringTask) 

# mergeTICLTask default for v5
mergeTICLTask = cms.Task(
    ticlLayerTileTask,
    ticlIterationsTask,
    ticlTracksterLinksTask
)


mtdSoATask = cms.Task(mtdSoA)
ticlCandidateTask = cms.Task(ticlCandidate)


if ticl_superclustering_mustache_ticl._isChosen():
    ticlIterLabels.append("ticlTracksterLinksSuperclusteringMustache")
else:
    ticlIterLabels.append("ticlTracksterLinksSuperclusteringDNN")




associatorsInstances = []
for labelts in ticlIterLabels:
    for labelsts in ['ticlSimTracksters', 'ticlSimTrackstersfromCPs']:
        associatorsInstances.append(labelts+'To'+labelsts)
        associatorsInstances.append(labelsts+'To'+labelts)

# iterTICLTask default for v5
iterTICLTask = cms.Task(
    mergeTICLTask,
    mtdSoATask, 
    ticlCandidateTask,
    ticlPFTask
)


# HFNose remains on legacy iterations
ticlLayerTileHFNose = ticlLayerTileProducer.clone(
    detector = 'HFNose'
)
ticlLayerTileHFNoseTask = cms.Task(ticlLayerTileHFNose)
iterHFNoseTICLTask = cms.Task(
    ticlLayerTileHFNoseTask,
    ticlHFNoseTrkEMStepTask,
    ticlHFNoseEMStepTask,
    ticlHFNoseTrkStepTask,
    ticlHFNoseHADStepTask,
    ticlHFNoseMIPStepTask
)
