import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorTasks.EBBeamCaloTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBBeamHodoTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBClusterTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBCosmicTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBIntegrityTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBLaserTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBOccupancyTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBPedestalOnlineTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBPedestalTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBStatusFlagsTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBTestPulseTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBTimingTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBTriggerTowerTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBSelectiveReadoutTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EBRawDataTask_cfi import *

ecalBarrelOccupancyTask.mergeRuns = True
ecalBarrelIntegrityTask.mergeRuns = True

ecalBarrelStatusFlagsTask.mergeRuns = True

ecalBarrelCosmicTask.mergeRuns = True
ecalBarrelLaserTask.mergeRuns = True
ecalBarrelPedestalOnlineTask.mergeRuns = True
ecalBarrelPedestalTask.mergeRuns = True
ecalBarrelTestPulseTask.mergeRuns = True

ecalBarrelTriggerTowerTask.mergeRuns = True
ecalBarrelTimingTask.mergeRuns = True

ecalBarrelBeamHodoTask.mergeRuns = True
ecalBarrelBeamCaloTask.mergeRuns = True

ecalBarrelClusterTask.mergeRuns = True

ecalBarrelSelectiveReadoutTask.mergeRuns = True
ecalBarrelRawDataTask.mergeRuns = True
