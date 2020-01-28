import FWCore.ParameterSet.Config as cms

from RecoMTD.TrackExtender.trackExtenderWithMTD_cfi import *
from RecoMTD.TimingIDTools.trackPUIDMVA_cfi import *

fastTimingGlobalRecoTask = cms.Task(trackExtenderWithMTD,trackPUIDMVA)
fastTimingGlobalReco = cms.Sequence(fastTimingGlobalRecoTask)
