import FWCore.ParameterSet.Config as cms

from DQM.EcalPreshowerMonitorModule.ESRawDataTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESIntegrityTask_cfi import *
#from DQM.EcalPreshowerMonitorModule.ESPedestalTask_cfi import *
from DQM.EcalPreshowerMonitorModule.ESOccupancyTask_cfi import *

#ecalPreshowerDefaultTasksSequence = cms.Sequence(ecalPreshowerOccupancyTask*ecalPreshowerPedestalTask)
ecalPreshowerDefaultTasksSequence = cms.Sequence(ecalPreshowerRawDataTask*ecalPreshowerIntegrityTask*ecalPreshowerOccupancyTask)

