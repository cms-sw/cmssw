from .hltTiclTracksterLinksL1Seeded_cfi import hltTiclTracksterLinksL1Seeded
import FWCore.ParameterSet.Config as cms


hltTiclTracksterLinksSuperclusteringMustacheL1Seeded = hltTiclTracksterLinksL1Seeded.clone(
    linkingPSet = cms.PSet(
        type = cms.string("SuperClusteringMustache"),
        algo_verbosity = cms.int32(0)
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHighL1Seeded")], 
)
