import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMDaqInfo_cfi import *
from DQM.DTMonitorClient.dtDAQInfo_cfi import *
from DQM.RPCMonitorClient.RPCDaqInfo_cfi import *
from DQM.EcalBarrelMonitorTasks.EBDaqInfoTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEDaqInfoTask_cfi import *
siStripDaqInfo = cms.EDFilter("SiStripDaqInfo")
cscDaqInfo = cms.EDFilter("CSCDaqInfo")
siPixelDaqInfo  = cms.EDFilter("SiPixelDaqInfo")

daq_dqmoffline = cms.Sequence(dqmDaqInfo*dtDAQInfo*rpcDaqInfo*ecalBarrelDaqInfoTask*ecalEndcapDaqInfoTask*siStripDaqInfo*cscDaqInfo*siPixelDaqInfo)

