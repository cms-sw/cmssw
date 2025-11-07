# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Standard TICL v5 iteration & singleton presets.

The algorithm parameter sets here are transcribed verbatim from the baseline
``RecoHGCal/TICL/python`` cff files (``CLUE3DHighStep_cff``, ``PRbyRecovery_cff``,
``superclustering_cff``, ``iterativeTICL_cff``).  pyTICL clones the same ``_cfi``
defaults and re-applies these overrides, so the generated config reproduces the
baseline byte-for-byte.  The *plumbing* (filtered_mask, seeding_regions,
tracksters_collections, masks, ticlCandidateSrc, iteration_label, itername) is
deliberately NOT encoded here -- it is computed by the assembler from the
iteration graph and checked by the validator.
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.model import TICLConfig, Global


# --------------------------------------------------------------------------- #
# Inference parameter sets (pattern-recognition stage)
# --------------------------------------------------------------------------- #

_CLUE3DHIGH_CNN = cms.PSet(
    algo_verbosity=cms.int32(0),
    type=cms.string("TracksterInferenceByCNN"),
    onnxModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/patternrecognition/id_v0.onnx"),
    inputNames=cms.vstring("input"),
    outputNames=cms.vstring("pid_output"),
    eid_min_cluster_energy=cms.double(1.0),
    eid_n_layers=cms.int32(50),
    eid_n_clusters=cms.int32(10),
    doPID=cms.int32(1),
    miniBatchSize=cms.untracked.int32(64),
)

_PR_DNN = cms.PSet(
    algo_verbosity=cms.int32(0),
    onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/patternrecognition/id_v0.onnx"),
    onnxEnergyModelPath=cms.string(""),
    inputNames=cms.vstring("input"),
    output_en=cms.vstring("enreg_output"),
    output_id=cms.vstring("pid_output"),
    eid_min_cluster_energy=cms.double(1),
    eid_n_layers=cms.int32(50),
    eid_n_clusters=cms.int32(10),
    doPID=cms.int32(1),
    doRegression=cms.int32(0),
    type=cms.string("TracksterInferenceByDNN"),
)

_PR_PFN = cms.PSet(
    algo_verbosity=cms.int32(0),
    onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/id_v0.onnx"),
    onnxEnergyModelPath=cms.string(""),
    inputNames=cms.vstring("input", "input_tr_features"),
    output_en=cms.vstring("enreg_output"),
    output_id=cms.vstring("pid_output"),
    eid_min_cluster_energy=cms.double(1),
    eid_n_layers=cms.int32(50),
    eid_n_clusters=cms.int32(10),
    doPID=cms.int32(1),
    doRegression=cms.int32(0),
    type=cms.string("TracksterInferenceByPFN"),
)

_RECOVERY_PFN = cms.PSet(
    algo_verbosity=cms.int32(0),
    onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/id_v0.onnx"),
    onnxEnergyModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/energy_v0.onnx"),
    inputNames=cms.vstring("input", "input_tr_features"),
    output_en=cms.vstring("enreg_output"),
    output_id=cms.vstring("pid_output"),
    eid_min_cluster_energy=cms.double(1),
    eid_n_layers=cms.int32(50),
    eid_n_clusters=cms.int32(10),
    doPID=cms.int32(0),
    doRegression=cms.int32(0),
    type=cms.string("TracksterInferenceByPFN"),
)


# --------------------------------------------------------------------------- #
# Per-iteration presets
# --------------------------------------------------------------------------- #

def apply_iteration_preset(it, name):
    """Populate an :class:`IterationSpec` ``it`` with the standard preset ``name``."""
    fn = _ITERATION_PRESETS.get(name)
    if fn is None:
        from RecoTICL.Configuration.model import PyTICLError
        raise PyTICLError("no iteration preset named %r (known: %s)"
                          % (name, ", ".join(sorted(_ITERATION_PRESETS))))
    fn(it)


def _preset_clue3dhigh(it):
    it.seeding_type = "SeedingRegionGlobal"
    it.filter_type = "ClusterFilterByAlgoAndSize"
    it.filter_params = dict(min_cluster_size=2)
    it.pattern_type = "CLUE3D"
    it.pattern_params = dict(
        criticalDensity=[0.6, 0.6, 0.6],
        criticalEtaPhiDistance=[0.025, 0.025, 0.025],
        kernelDensityFactor=[0.2, 0.2, 0.2],
        algo_verbosity=0,
        doPidCut=True,
        cutHadProb=999,
    )
    it.trackster_extra = dict(
        inferenceAlgo=cms.string("TracksterInferenceByCNN"),
        pluginInferenceAlgoTracksterInferenceByCNN=_CLUE3DHIGH_CNN.clone(),
        pluginInferenceAlgoTracksterInferenceByDNN=_PR_DNN.clone(),
        pluginInferenceAlgoTracksterInferenceByPFN=_PR_PFN.clone(),
    )


def _preset_recovery(it):
    it.seeding_type = "SeedingRegionGlobal"
    it.filter_type = "ClusterFilterByAlgoAndSize"
    it.filter_params = dict(min_cluster_size=2, algo_number=[6, 7, 8])
    it.pattern_type = "Recovery"
    it.pattern_params = dict(algo_verbosity=0)
    it.masks_from = "CLUE3DHigh"
    it.trackster_extra = dict(
        inferenceAlgo=cms.string(""),
        pluginInferenceAlgoTracksterInferenceByPFN=_RECOVERY_PFN.clone(),
    )


_ITERATION_PRESETS = {
    "CLUE3DHigh": _preset_clue3dhigh,
    "Recovery": _preset_recovery,
}


# --------------------------------------------------------------------------- #
# Singleton-stage default overrides (links, superclustering, candidate, pf)
# --------------------------------------------------------------------------- #

def links_defaults():
    """Standard ``ticlTracksterLinks`` overrides (Skeletons linking)."""
    return dict(
        linkingPSet=cms.PSet(
            cylinder_radius_sqr_split=cms.double(9),
            proj_distance_split=cms.double(5),
            track_time_quality_threshold=cms.double(0.5),
            min_num_lcs=cms.uint32(15),
            min_trackster_energy=cms.double(20),
            pca_quality_th=cms.double(0.85),
            dot_prod_th=cms.double(0.97),
            lower_boundary=cms.vdouble(20, 10),
            upper_boundary=cms.vdouble(150, 100),
            upper_distance_projective_sqr=cms.vdouble(4, 60),
            lower_distance_projective_sqr=cms.vdouble(4, 60),
            min_distance_z=cms.vdouble(35, 35),
            upper_distance_projective_sqr_closest_points=cms.vdouble(5, 30),
            lower_distance_projective_sqr_closest_points=cms.vdouble(10, 50),
            max_z_distance_closest_points=cms.vdouble(35, 35),
            cylinder_radius_sqr=cms.vdouble(9, 15),
            deltaRxy=cms.double(4.0),
            algo_verbosity=cms.int32(0),
            type=cms.string("Skeletons"),
        ),
        regressionAndPid=cms.bool(False),
        inferenceAlgo=cms.string(""),
        pluginInferenceAlgoTracksterInferenceByDNN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            inputNames=cms.vstring("input"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            eid_min_cluster_energy=cms.double(1),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            onnxEnergyModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/linking/energy_v0.onnx"),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/linking/id_v0.onnx"),
            type=cms.string("TracksterInferenceByDNN"),
        ),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            inputNames=cms.vstring("input", "input_tr_features"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            eid_min_cluster_energy=cms.double(2.5),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            onnxEnergyModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx"),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx"),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def supercluster_dnn_defaults():
    """Standard ``ticlTracksterLinksSuperclusteringDNN`` overrides."""
    return dict(
        linkingPSet=cms.PSet(
            type=cms.string("SuperClusteringDNN"),
            algo_verbosity=cms.int32(0),
            onnxModelPath=cms.string("RecoHGCal/TICL/data/superclustering/supercls_v3.onnx"),
            nnWorkingPoint=cms.double(0.57247),
        ),
    )


def candidate_defaults():
    """Standard ``ticlCandidate`` overrides (PFN inference + regression/PID)."""
    return dict(
        inferenceAlgo=cms.string("TracksterInferenceByPFN"),
        regressionAndPid=cms.bool(True),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx"),
            onnxEnergyModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx"),
            inputNames=cms.vstring("input", "input_tr_features"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            eid_min_cluster_energy=cms.double(2.5),
            eid_n_layers=cms.int32(50),
            eid_n_clusters=cms.int32(10),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def pf_defaults():
    """Standard ``pfTICL`` overrides."""
    return dict(useTimingAverage=cms.bool(True))


# --------------------------------------------------------------------------- #
# Full v5 default configuration
# --------------------------------------------------------------------------- #

def v5(name="v5"):
    """Return a :class:`TICLConfig` reproducing the default ``iterTICLTask`` (v5)."""
    cfg = (TICLConfig(name)
           .iteration("CLUE3DHigh").preset()
           .iteration("Recovery").preset()
           .links(["CLUE3DHigh", "Recovery"], **links_defaults())
           .superclustering_dnn(source="CLUE3DHigh", **supercluster_dnn_defaults())
           .candidate(**candidate_defaults())
           .pf(**pf_defaults()))
    return cfg
