import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtDAQInfo_cfi import *

dtCertification = cms.Sequence(dtDAQInfo)

