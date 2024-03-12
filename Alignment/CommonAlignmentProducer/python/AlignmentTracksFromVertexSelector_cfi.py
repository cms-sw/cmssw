import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.alignmentTrackFromVertexSelectorModule_cfi import alignmentTrackFromVertexSelectorModule
AlignmentTracksFromVertexSelector = alignmentTrackFromVertexSelectorModule.clone(src = cms.InputTag("generalTracks"),
                                                                                 vertices = cms.InputTag("offlinePrimaryVertices"),
                                                                                 vertexIndex = cms.uint32(0),
                                                                                 filter = cms.bool(False))
# foo bar baz
# 0Wm3xbnCxOhdu
