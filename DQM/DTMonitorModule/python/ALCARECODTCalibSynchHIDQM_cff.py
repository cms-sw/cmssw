import FWCore.ParameterSet.Config as cms

from DQMOffline.CalibMuon.dtPreCalibrationTask_cfi import *
from DQM.DTMonitorModule.dtResolutionTask_cfi import *
from DQM.DTMonitorModule.dtTriggerSynchTask_cfi import *

dtPreCalibrationTaskAlcaHI = dtPreCalibTask.clone(
   SaveFile = False,
   folderName = 'AlCaReco/DtCalibSynchHI/01-Calibration'
)

dtAlcaResolutionMonitorHI = dtResolutionAnalysisMonitor.clone(
  topHistoFolder = 'AlCaReco/DtCalibSynchHI/01-Calibration'
)
 
dtTriggerSynchMonitorHI = dtTriggerSynchMonitor.clone(
  baseDir = 'AlCaReco/DtCalibSynchHI/02-Synchronization',             
  SEGInputTag = 'dt4DSegmentsNoWire',             
  rangeWithinBX  = False,
  nBXHigh        = 3,
  nBXLow         = -2
)

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
