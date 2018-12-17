import FWCore.ParameterSet.Config as cms

from RecoMTD.TrackExtender.trackExtenderWithMTD_cfi import *

fastTimingGlobalReco = cms.Sequence(trackExtenderWithMTD)
