import FWCore.ParameterSet.Config as cms


from DQM.EcalPreshowerMonitorModule.ESPedestalTask_cfi import *


ecalPreshowerDefaultTasksSequence = cms.Sequence(ecalPreshowerPedestalTask)

