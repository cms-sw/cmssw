import FWCore.ParameterSet.Config as cms

PureTracks = cms.EDFilter("TrackSelector",
                       src = cms.InputTag("hiGeneralTracks"),
                       cut = cms.string('quality("highPurity")'))

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
offlinePrimaryVertices.TrackLabel = 'PureTracks'
