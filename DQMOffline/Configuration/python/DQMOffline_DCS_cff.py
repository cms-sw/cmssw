import FWCore.ParameterSet.Config as cms

siStripDcsInfo = cms.EDAnalyzer("SiStripDcsInfo")
from DQM.SiPixelCommon.SiPixelOfflineDQM_client_cff import *
from DQM.EcalBarrelMonitorTasks.EcalDcsInfoTask_cfi import *
from DQM.DTMonitorClient.dtDCSByLumiSummary_cfi import *
from DQM.HcalMonitorClient.HcalDCSInfo_cfi import *
from DQM.RPCMonitorClient.RPCDCSSummary_cfi import *
from DQM.CSCMonitorModule.csc_dcs_info_cfi import *
from DQM.EcalPreshowerMonitorModule.ESDcsInfoTask_cfi import *

dcs_dqmoffline = cms.Sequence(siStripDcsInfo*sipixelDcsInfo*ecalDcsInfoTask*dtDCSByLumiSummary*hcalDCSInfo*rpcDCSSummary*cscDcsInfo*ecalPreshowerDcsInfoTask)

