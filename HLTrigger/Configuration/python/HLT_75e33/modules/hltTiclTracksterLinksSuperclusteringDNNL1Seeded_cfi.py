from .hltTiclTracksterLinksL1Seeded_cfi import hltTiclTracksterLinksL1Seeded
import FWCore.ParameterSet.Config as cms

hltTiclTracksterLinksSuperclusteringDNNL1Seeded = hltTiclTracksterLinksL1Seeded.clone(
    linkingPSet = cms.PSet(
        type=cms.string("SuperClusteringDNN"),
        algo_verbosity=cms.int32(0),
        onnxModelPath = cms.string("RecoHGCal/TICL/data/superclustering/supercls_v3.onnx"),
        nnWorkingPoint=cms.float(0.57247),
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHighL1Seeded")], # to be changed to ticlTrackstersCLUE3DEM once separate CLUE3D iterations are introduced
)