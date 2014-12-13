import FWCore.ParameterSet.Config as cms

from DQMOffline.CalibMuon.dtPreCalibrationTask_cfi import *
from DQM.DTMonitorModule.dtResolutionTask_cfi import *
from DQM.DTMonitorModule.dtTriggerSynchTask_cfi import *

dtPreCalibrationTaskAlcaHI = dtPreCalibTask.clone()
dtPreCalibrationTaskAlcaHI.SaveFile = False
dtPreCalibrationTaskAlcaHI.folderName = 'AlCaReco/DtCalibSynchHI/01-Calibration'

dtAlcaResolutionMonitorHI = dtResolutionAnalysisMonitor.clone()
dtAlcaResolutionMonitorHI.topHistoFolder = "AlCaReco/DtCalibSynchHI/01-Calibration"
 
dtTriggerSynchMonitorHI = dtTriggerSynchMonitor.clone()
dtTriggerSynchMonitorHI.baseDir = 'AlCaReco/DtCalibSynchHI/02-Synchronization'             
dtTriggerSynchMonitorHI.SEGInputTag = 'dt4DSegmentsNoWire'             
dtTriggerSynchMonitorHI.rangeWithinBX  = False
dtTriggerSynchMonitorHI.nBXHigh        = 3
dtTriggerSynchMonitorHI.nBXLow         = -2

#from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from CalibMuon.DTCalibration.ALCARECODtCalibHI_cff import ALCARECODtCalibHIHLTFilter
#ALCARECODtCalibHIHLTDQM = hltMonBitSummary.clone(
#    directory = 'AlCaReco/DtCalibSynchHI/HLTSummary',
#    histLabel = 'DtCalibSynchHI',
#    HLTPaths = ["HLT_.*Mu.*"],
#    eventSetupPathsKey =  ALCARECODtCalibHIHLTFilter.eventSetupPathsKey.value()
#)

#ALCARECODTCalibSynchHIDQM = cms.Sequence( dtPreCalibrationTaskAlcaHI +
#                                          dtAlcaResolutionMonitorHI + 
#                                          ALCARECODtCalibHIHLTDQM )
ALCARECODTCalibSynchHIDQM = cms.Sequence( dtPreCalibrationTaskAlcaHI +
                                          dtAlcaResolutionMonitorHI )
