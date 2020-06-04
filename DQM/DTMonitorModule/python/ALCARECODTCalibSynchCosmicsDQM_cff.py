import FWCore.ParameterSet.Config as cms

from DQMOffline.CalibMuon.dtPreCalibrationTask_cfi import *
from DQM.DTMonitorModule.dtResolutionTask_cfi import *
from DQM.DTMonitorModule.dtTriggerSynchTask_cfi import *

dtPreCalibrationTaskAlcaCosmics = dtPreCalibTask.clone()
dtPreCalibrationTaskAlcaCosmics.SaveFile = False
dtPreCalibrationTaskAlcaCosmics.folderName = 'AlCaReco/DtCalibSynchCosmics/01-Calibration'

dtAlcaResolutionMonitorCosmics = dtResolutionAnalysisMonitor.clone()
dtAlcaResolutionMonitorCosmics.topHistoFolder = "AlCaReco/DtCalibSynchCosmics/01-Calibration"
 
#dtTriggerSynchMonitorCosmics = dtTriggerSynchMonitor.clone()
#dtTriggerSynchMonitorCosmics.baseDir = 'AlCaReco/DtCalibSynchCosmics/02-Synchronization'             
#dtTriggerSynchMonitorCosmics.SEGInputTag = 'dt4DSegments'             
#dtTriggerSynchMonitorCosmics.rangeWithinBX  = False
#dtTriggerSynchMonitorCosmics.nBXHigh        = 3
#dtTriggerSynchMonitorCosmics.nBXLow         = -2

#from DQM.HLTEvF.HLTMonBitSummary_cfi import hltMonBitSummary
from CalibMuon.DTCalibration.ALCARECODtCalibCosmics_cff import ALCARECODtCalibCosmicsHLTFilter
#ALCARECODtCalibCosmicsHLTDQM = hltMonBitSummary.clone(
#    directory = 'AlCaReco/DtCalibSynchCosmics/HLTSummary',
#    histLabel = 'DtCalibSynchCosmics',
#    HLTPaths = ["HLT_.*Mu.*"],
#    eventSetupPathsKey =  ALCARECODtCalibCosmicsHLTFilter.eventSetupPathsKey.value()
#)

#ALCARECODTCalibSynchCosmicsDQM = cms.Sequence( dtPreCalibrationTaskAlcaCosmics +
#                                               dtAlcaResolutionMonitorCosmics + 
#                                               ALCARECODtCalibCosmicsHLTDQM )

ALCARECODTCalibSynchCosmicsDQM = cms.Sequence( dtPreCalibrationTaskAlcaCosmics +
                                               dtAlcaResolutionMonitorCosmics )

