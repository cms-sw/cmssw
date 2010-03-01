import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtDAQInfo_cfi import *
from DQM.RPCMonitorClient.RPCDaqInfo_cfi import *
from DQM.EcalBarrelMonitorTasks.EBDaqInfoTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEDaqInfoTask_cfi import *
siStripDaqInfo = cms.EDAnalyzer("SiStripDaqInfo")
from DQM.CSCMonitorModule.csc_daq_info_cfi import *
siPixelDaqInfo  = cms.EDAnalyzer("SiPixelDaqInfo")
from DQM.HcalMonitorClient.HcalDAQInfo_cfi import *
from DQM.RPCMonitorClient.RPCDaqInfo_cfi import *
from DQM.EcalPreshowerMonitorModule.ESDaqInfoTask_cfi import *

daq_dqmoffline = cms.Sequence(dtDAQInfo*rpcDaqInfo*ecalBarrelDaqInfoTask*ecalEndcapDaqInfoTask*siStripDaqInfo*cscDaqInfo*siPixelDaqInfo*hcalDAQInfo*rpcDaqInfo*ecalPreshowerDaqInfoTask)

