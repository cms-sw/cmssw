import FWCore.ParameterSet.Config as cms

from RecoMTD.TrackExtender.trackExtenderWithMTD_cfi import *

fastTimingGlobalRecoTask = cms.Task(trackExtenderWithMTD)
fastTimingGlobalReco = cms.Sequence(fastTimingGlobalRecoTask)
