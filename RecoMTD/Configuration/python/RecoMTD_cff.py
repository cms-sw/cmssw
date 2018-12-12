import FWCore.ParameterSet.Config as cms

from RecoMTD.TrackExtender.trackExtender_cfi import *

fastTimingGlobalReco = cms.Sequence(mtdTrackExtender)
