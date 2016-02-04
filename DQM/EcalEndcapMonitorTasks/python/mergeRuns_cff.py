import FWCore.ParameterSet.Config as cms

from DQM.EcalEndcapMonitorTasks.EEBeamCaloTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEBeamHodoTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEClusterTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EECosmicTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEIntegrityTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EELaserTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EELedTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEOccupancyTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEPedestalOnlineTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEPedestalTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EEStatusFlagsTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EETestPulseTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EETimingTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EETriggerTowerTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EESelectiveReadoutTask_cfi import *
from DQM.EcalEndcapMonitorTasks.EERawDataTask_cfi import *

ecalEndcapOccupancyTask.mergeRuns = True
ecalEndcapIntegrityTask.mergeRuns = True

ecalEndcapStatusFlagsTask.mergeRuns = True

ecalEndcapCosmicTask.mergeRuns = True
ecalEndcapLaserTask.mergeRuns = True
ecalEndcapLedTask.mergeRuns = True
ecalEndcapPedestalOnlineTask.mergeRuns = True
ecalEndcapPedestalTask.mergeRuns = True
ecalEndcapTestPulseTask.mergeRuns = True

ecalEndcapTriggerTowerTask.mergeRuns = True
ecalEndcapTimingTask.mergeRuns = True

ecalEndcapBeamHodoTask.mergeRuns = True
ecalEndcapBeamCaloTask.mergeRuns = True

ecalEndcapClusterTask.mergeRuns = True

ecalEndcapSelectiveReadoutTask.mergeRuns = True
ecalEndcapRawDataTask.mergeRuns = True
