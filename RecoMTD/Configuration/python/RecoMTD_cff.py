import FWCore.ParameterSet.Config as cms

from RecoMTD.TrackExtender.trackExtender_cfi import mtdTrackExtender

fastTimingGlobalReco = cms.Sequence(mtdTrackExtender)
