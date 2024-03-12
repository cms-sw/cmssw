import FWCore.ParameterSet.Config as cms

from RecoMTD.TrackExtender.trackExtenderWithMTD_cfi import *
from RecoMTD.TimingIDTools.mtdTrackQualityMVA_cfi import *

fastTimingGlobalRecoTask = cms.Task(trackExtenderWithMTD,mtdTrackQualityMVA)
fastTimingGlobalReco = cms.Sequence(fastTimingGlobalRecoTask)
# foo bar baz
# uwb5LAvTKk5EH
# 9C7V358PDnq1y
