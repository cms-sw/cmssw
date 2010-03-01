import FWCore.ParameterSet.Config as cms

siStripDcsInfo = cms.EDAnalyzer("SiStripDcsInfo")
siPixelDcsInfo = cms.EDAnalyzer("SiPixelDcsInfo")
from DQM.EcalBarrelMonitorTasks.EBDcsInfoTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEDcsInfoTask_cfi import *
from DQM.DTMonitorClient.dtDCSSummary_cfi import *
from DQM.HcalMonitorClient.HcalDCSInfo_cfi import *
from DQM.RPCMonitorClient.RPCDCSSummary_cfi import *
from DQM.CSCMonitorModule.csc_dcs_info_cfi import *
from DQM.EcalPreshowerMonitorModule.ESDcsInfoTask_cfi import *

dcs_dqmoffline = cms.Sequence(siStripDcsInfo*siPixelDcsInfo*ecalBarrelDcsInfoTask*ecalEndcapDcsInfoTask*dtDCSSummary*hcalDCSInfo*rpcDCSSummary*cscDcsInfo*ecalPreshowerDcsInfoTask)

