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

ecalEndcapOccupancyTask.enableCleanup = True
ecalEndcapIntegrityTask.enableCleanup = True

ecalEndcapStatusFlagsTask.enableCleanup = True

ecalEndcapCosmicTask.enableCleanup = True
ecalEndcapLaserTask.enableCleanup = True
ecalEndcapLedTask.enableCleanup = True
ecalEndcapPedestalOnlineTask.enableCleanup = True
ecalEndcapPedestalTask.enableCleanup = True
ecalEndcapTestPulseTask.enableCleanup = True

ecalEndcapTriggerTowerTask.enableCleanup = True
ecalEndcapTimingTask.enableCleanup = True

ecalEndcapBeamHodoTask.enableCleanup = True
ecalEndcapBeamCaloTask.enableCleanup = True

ecalEndcapClusterTask.enableCleanup = True

ecalEndcapSelectiveReadoutTask.enableCleanup = True
ecalEndcapRawDataTask.enableCleanup = True
