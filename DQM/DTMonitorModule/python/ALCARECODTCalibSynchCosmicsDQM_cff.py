import FWCore.ParameterSet.Config as cms

from DQMOffline.CalibMuon.dtPreCalibrationTask_cfi import *
from DQM.DTMonitorModule.dtResolutionTask_cfi import *
from DQM.DTMonitorModule.dtTriggerSynchTask_cfi import *

dtPreCalibrationTaskAlcaCosmics = dtPreCalibTask.clone(
  SaveFile = False,
  folderName = 'AlCaReco/DtCalibSynchCosmics/01-Calibration'
)

dtAlcaResolutionMonitorCosmics = dtResolutionAnalysisMonitor.clone(
  topHistoFolder = 'AlCaReco/DtCalibSynchCosmics/01-Calibration'
)

#dtTriggerSynchMonitorCosmics = dtTriggerSynchMonitor.clone(
#  baseDir = 'AlCaReco/DtCalibSynchCosmics/02-Synchronization',
#  SEGInputTag = 'dt4DSegments',
#  rangeWithinBX  = False,
#  nBXHigh        = 3,
#  nBXLow         = -2
# )

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

