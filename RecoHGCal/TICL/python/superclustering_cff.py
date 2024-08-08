import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.tracksterLinksProducer_cfi import tracksterLinksProducer as _tracksterLinksProducer
from RecoHGCal.TICL.ticlEGammaSuperClusterProducer_cfi import ticlEGammaSuperClusterProducer
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import particleFlowSuperClusterHGCal

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5
from Configuration.ProcessModifiers.ticl_superclustering_dnn_cff import ticl_superclustering_dnn
from Configuration.ProcessModifiers.ticl_superclustering_mustache_pf_cff import ticl_superclustering_mustache_pf
from Configuration.ProcessModifiers.ticl_superclustering_mustache_ticl_cff import ticl_superclustering_mustache_ticl

ticlTracksterLinksSuperclusteringDNN = _tracksterLinksProducer.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.FileInPath("RecoHGCal/TICL/data/superclustering/supercls_v2p1.onnx"),
        nnWorkingPoint=cms.double(0.3),
    ),
    tracksters_collections = [cms.InputTag("ticlTrackstersCLUE3DHigh")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

ticlTracksterLinksSuperclusteringMustache = _tracksterLinksProducer.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringMustache"),
        algo_verbosity=cms.int32(0)
    ),
    tracksters_collections = [cms.InputTag("ticlTrackstersCLUE3DHigh")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)

### Superclustering : 3 options : DNN, Mustache-TICL (from tracksters), Mustache-PF (converting tracksters to PFClusters, default for ticl_v4, enable with modifier for v5)
ticlSuperclusteringTask = cms.Task()

# DNN
_dnn_task = cms.Task(ticlTracksterLinksSuperclusteringDNN)
ticl_superclustering_dnn.toReplaceWith(ticlSuperclusteringTask, _dnn_task)
ticl_superclustering_dnn.toModify(ticlEGammaSuperClusterProducer, ticlSuperClusters=cms.InputTag("ticlTracksterLinksSuperclusteringDNN"))
ticl_superclustering_dnn.toReplaceWith(particleFlowSuperClusterHGCal, ticlEGammaSuperClusterProducer)

# Mustache-TICL
_mustache_ticl_task = cms.Task(ticlTracksterLinksSuperclusteringMustache)
ticl_superclustering_mustache_ticl.toReplaceWith(ticlSuperclusteringTask, _mustache_ticl_task)
ticl_superclustering_mustache_ticl.toModify(ticlEGammaSuperClusterProducer, ticlSuperClusters=cms.InputTag("ticlTracksterLinksSuperclusteringMustache"))
ticl_superclustering_mustache_ticl.toReplaceWith(particleFlowSuperClusterHGCal, ticlEGammaSuperClusterProducer)

# Mustache-PF
# (no changes to make)
