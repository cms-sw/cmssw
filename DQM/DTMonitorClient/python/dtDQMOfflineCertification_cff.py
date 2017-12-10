import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtDAQInfo_cfi import *
from DQM.DTMonitorClient.dtDCSByLumiSummary_cfi import *
from DQM.DTMonitorClient.dtCertificationSummary_cfi import *

dtCertification = cms.Sequence(dtDAQInfo + dtDCSByLumiSummary + dtCertificationSummary)

#from Configuration.Eras.Modifier_run2_DT_2018_cff import run2_DT_2018
#run2_DT_2018.toModify(dtDAQInfo,checkUros  = cms.untracked.bool(True))

