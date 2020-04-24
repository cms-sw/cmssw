import FWCore.ParameterSet.Config as cms

from DQMOffline.CalibMuon.dtPreCalibrationTask_cfi import *
from DQM.DTMonitorModule.dtResolutionTask_cfi import *
from DQM.DTMonitorModule.dtTriggerSynchTask_cfi import *

dtPreCalibrationTaskAlca = dtPreCalibTask.clone()
dtPreCalibrationTaskAlca.SaveFile = False
dtPreCalibrationTaskAlca.folderName = 'AlCaReco/DtCalibSynch/01-Calibration'

dtAlcaResolutionMonitor = dtResolutionAnalysisMonitor.clone()
dtAlcaResolutionMonitor.topHistoFolder = "AlCaReco/DtCalibSynch/01-Calibration"
 
dtTriggerSynchMonitor.baseDir = 'AlCaReco/DtCalibSynch/02-Synchronization'             
dtTriggerSynchMonitor.SEGInputTag = 'dt4DSegmentsNoWire'             
dtTriggerSynchMonitor.rangeWithinBX  = False
dtTriggerSynchMonitor.nBXHigh        = 3
dtTriggerSynchMonitor.nBXLow         = -2

#from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from CalibMuon.DTCalibration.ALCARECODtCalib_cff import ALCARECODtCalibHLTFilter
#ALCARECODtCalibHLTDQM = hltMonBitSummary.clone(
#    directory = 'AlCaReco/DtCalibSynch/HLTSummary',
#    histLabel = 'DtCalibSynch',
#    HLTPaths = ["HLT_.*Mu.*"],
#    eventSetupPathsKey =  ALCARECODtCalibHLTFilter.eventSetupPathsKey.value()
#)

#ALCARECODTCalibSynchDQM = cms.Sequence( dtPreCalibrationTaskAlca +
#                                        dtAlcaResolutionMonitor + 
#                                        dtTriggerSynchMonitor +
#                                        ALCARECODtCalibHLTDQM )
ALCARECODTCalibSynchDQM = cms.Sequence( dtPreCalibrationTaskAlca +
                                        dtAlcaResolutionMonitor +
                                        dtTriggerSynchMonitor )
