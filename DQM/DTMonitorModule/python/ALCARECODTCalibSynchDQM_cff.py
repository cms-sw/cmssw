import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorModule.dtResolutionTask_cfi import *
from DQM.DTMonitorModule.dtTriggerSynchTask_cfi import *

dtResolutionAnalysisMonitor.topHistoFolder = "DtCalibSynch/01-Calibration" 
dtTriggerSynchMonitor.baseDir = 'DtCalibSynch/02-Synchronization'             
dtTriggerSynchMonitor.SEGInputTag = 'dt4DSegmentsNoWire'             
dtTriggerSynchMonitor.rangeWithinBX  = False
dtTriggerSynchMonitor.nBXHigh        = 3
dtTriggerSynchMonitor.nBXLow         = -2

ALCARECODTCalibSynchDQM = cms.Sequence( dtResolutionAnalysisMonitor + dtTriggerSynchMonitor )
