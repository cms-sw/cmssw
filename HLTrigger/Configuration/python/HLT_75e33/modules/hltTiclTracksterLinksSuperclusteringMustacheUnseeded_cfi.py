from .hltTiclTracksterLinks_cfi import hltTiclTracksterLinks
import FWCore.ParameterSet.Config as cms


hltTiclTracksterLinksSuperclusteringMustacheUnseeded = hltTiclTracksterLinks.clone(
    linkingPSet = cms.PSet(
        type = cms.string("SuperClusteringMustache"),
        algo_verbosity = cms.int32(0)
    ),
    tracksters_collections = [cms.InputTag("hltTiclTrackstersCLUE3DHigh")], 
)
