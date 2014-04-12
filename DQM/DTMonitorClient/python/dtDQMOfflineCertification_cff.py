import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtDAQInfo_cfi import *
from DQM.DTMonitorClient.dtDCSByLumiSummary_cfi import *
from DQM.DTMonitorClient.dtCertificationSummary_cfi import *

dtCertification = cms.Sequence(dtDAQInfo + dtDCSByLumiSummary + dtCertificationSummary)

