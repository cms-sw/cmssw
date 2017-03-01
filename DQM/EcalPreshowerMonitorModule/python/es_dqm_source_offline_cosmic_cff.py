import FWCore.ParameterSet.Config as cms

from DQM.EcalPreshowerMonitorModule.ESRawDataTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESIntegrityTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESFEDIntegrityTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESOccupancyTask_cfi import *

dqmInfoES = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalPreshower')
)

es_dqm_source_offline = cms.Sequence(ecalPreshowerRawDataTask*ecalPreshowerFEDIntegrityTask*ecalPreshowerIntegrityTask*ecalPreshowerOccupancyTask)
