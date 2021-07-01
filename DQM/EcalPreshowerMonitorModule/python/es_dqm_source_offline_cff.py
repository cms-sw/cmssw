import FWCore.ParameterSet.Config as cms

from DQM.EcalPreshowerMonitorModule.ESRawDataTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESIntegrityTask_cfi import *
#Check if perLSsaving is enabled to mask MEs vs LS
from DQMServices.Core.DQMStore_cfi import DQMStore
if(not DQMStore.saveByLumi):
    ecalPreshowerIntegrityTask.DoLumiAnalysis = True
from DQM.EcalPreshowerMonitorModule.ESFEDIntegrityTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESOccupancyTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESTrendTask_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoES = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('EcalPreshower')
)

es_dqm_source_offline = cms.Sequence(ecalPreshowerRawDataTask*ecalPreshowerFEDIntegrityTask*ecalPreshowerIntegrityTask*ecalPreshowerOccupancyTask*ecalPreshowerTrendTask)
