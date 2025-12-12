# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Phase-2 HLT presets for the TICL chain (HLTIterTICLSequence).

The HLT iterations (CLUE3DHigh, Recovery) share the offline algorithm presets;
only the labels and inputs differ (handled by the HLT :class:`Target`).  The
singletons (links / candidate / pf) are HLT-tuned and defined here.  pyTICL
reproduces ``HLTIterTICLSequence`` byte-for-byte (see test_reproduce_hlt.py).
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.model import TICLConfig


def hlt_links_defaults():
    """Overrides for hltTiclTracksterLinks (HLT Skeletons linking, empty ONNX)."""
    return dict(
        regressionAndPid=cms.bool(False),
        inferenceAlgo=cms.string(""),
        linkingPSet=cms.PSet(
            algo_verbosity=cms.int32(0),
            cylinder_radius_sqr=cms.vdouble(9, 15),
            cylinder_radius_sqr_split=cms.double(9),
            deltaRxy=cms.double(4),
            dot_prod_th=cms.double(0.97),
            lower_boundary=cms.vdouble(20, 10),
            lower_distance_projective_sqr=cms.vdouble(4, 60),
            lower_distance_projective_sqr_closest_points=cms.vdouble(10, 50),
            max_z_distance_closest_points=cms.vdouble(35, 35),
            min_distance_z=cms.vdouble(35, 35),
            min_num_lcs=cms.uint32(15),
            min_trackster_energy=cms.double(20),
            onnxModelPath=cms.string(""),
            pca_quality_th=cms.double(0.85),
            proj_distance_split=cms.double(5),
            track_time_quality_threshold=cms.double(0.5),
            type=cms.string("Skeletons"),
            upper_boundary=cms.vdouble(150, 100),
            upper_distance_projective_sqr=cms.vdouble(4, 60),
            upper_distance_projective_sqr_closest_points=cms.vdouble(5, 30),
        ),
        pluginInferenceAlgoTracksterInferenceByDNN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            eid_min_cluster_energy=cms.double(1),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input"),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string(""),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByDNN"),
        ),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            eid_min_cluster_energy=cms.double(1),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input", "input_tr_features"),
            miniBatchSize=cms.untracked.int32(64),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string(""),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def hlt_clue3dhigh_inference():
    """HLT CLUE3DHigh pattern-recognition DNN/PFN PSets (differ from offline:
    no eid_min_cluster_energy in DNN, miniBatchSize added in PFN)."""
    return dict(
        pluginInferenceAlgoTracksterInferenceByDNN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(0),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input"),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/DNN/patternrecognition/id_v0.onnx"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByDNN"),
        ),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(0),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input", "input_tr_features"),
            miniBatchSize=cms.untracked.int32(64),
            onnxEnergyModelPath=cms.string(""),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/patternrecognition/id_v0.onnx"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def hlt_candidate_defaults():
    """Overrides for hltTiclCandidate (PFN inference + HLT track/muon inputs)."""
    return dict(
        inferenceAlgo=cms.string("TracksterInferenceByPFN"),
        regressionAndPid=cms.bool(True),
        tracks=cms.InputTag("hltGeneralTracks"),
        muons=cms.InputTag("hltPhase2L3Muons"),
        useMTDTiming=cms.bool(False),
        useTimingAverage=cms.bool(False),
        pluginInferenceAlgoTracksterInferenceByPFN=cms.PSet(
            algo_verbosity=cms.int32(0),
            doPID=cms.int32(1),
            doRegression=cms.int32(1),
            eid_min_cluster_energy=cms.double(2.5),
            eid_n_clusters=cms.int32(10),
            eid_n_layers=cms.int32(50),
            inputNames=cms.vstring("input", "input_tr_features"),
            onnxEnergyModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/PFN/linking/energy_v1.onnx"),
            onnxPIDModelPath=cms.string("RecoHGCal/TICL/data/ticlv5/onnx_models/CNN/linking/id_v0.onnx"),
            output_en=cms.vstring("enreg_output"),
            output_id=cms.vstring("pid_output"),
            type=cms.string("TracksterInferenceByPFN"),
        ),
    )


def hlt_pf_defaults():
    """Overrides for hltPfTICL (HLT muon source, no MTD timing)."""
    return dict(muonSrc=cms.InputTag("hltPhase2L3Muons"), useMTDTiming=cms.bool(False))


def v5_hlt(name="v5_hlt"):
    """Return a :class:`TICLConfig` reproducing HLTIterTICLSequence (9 modules)."""
    cfg = (TICLConfig(name, target="HLT")
           .iteration("CLUE3DHigh").preset().trackster_params(**hlt_clue3dhigh_inference())
           .iteration("Recovery").preset().masks_from("CLUE3DHigh")
           .links(["CLUE3DHigh", "Recovery"], **hlt_links_defaults())
           .candidate(**hlt_candidate_defaults())
           .pf(**hlt_pf_defaults()))
    cfg.include_mtd = False   # the HLT iterTICL sequence has no mtdSoA stage
    return cfg
