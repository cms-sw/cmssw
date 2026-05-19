from .hltTiclTracksterLinks_cfi import hltTiclTracksterLinks
import FWCore.ParameterSet.Config as cms

hltTiclTracksterLinksSuperclusteringDNNUnseeded = hltTiclTracksterLinks.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.string("RecoHGCal/TICL/data/superclustering/supercls_v3.onnx"),
        nnWorkingPoint=cms.double(0.57247),
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHigh")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)