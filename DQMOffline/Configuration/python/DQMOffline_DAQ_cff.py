import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtDAQInfo_cfi import *
from DQM.RPCMonitorClient.RPCDaqInfo_cfi import *
from DQM.EcalBarrelMonitorTasks.EcalDaqInfoTask_cfi import *
siStripDaqInfo = cms.EDAnalyzer("SiStripDaqInfo")
from DQM.CSCMonitorModule.csc_daq_info_cfi import *
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.HcalMonitorClient.HcalDAQInfo_cfi import *
from DQM.RPCMonitorClient.RPCDaqInfo_cfi import *
from DQM.EcalPreshowerMonitorModule.ESDaqInfoTask_cfi import *

daq_dqmoffline = cms.Sequence(dtDAQInfo*rpcDaqInfo*ecalDaqInfoTask*siStripDaqInfo*cscDaqInfo*sipixelDaqInfo*hcalDAQInfo*ecalPreshowerDaqInfoTask)

