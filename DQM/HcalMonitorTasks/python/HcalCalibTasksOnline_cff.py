import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorTasks.HcalDigiMonitor_cfi          import *
from DQM.HcalMonitorTasks.HcalHotCellMonitor_cfi       import *
from DQM.HcalMonitorTasks.HcalDeadCellMonitor_cfi      import *
from DQM.HcalMonitorTasks.HcalRecHitMonitor_cfi        import *
from DQM.HcalMonitorTasks.HcalNZSMonitor_cfi           import *
from DQM.HcalMonitorTasks.HcalBeamMonitor_cfi          import *
from DQM.HcalMonitorTasks.HcalRawDataMonitor_cfi       import *
from DQM.HcalMonitorTasks.HcalTrigPrimMonitor_cfi      import *
from DQM.HcalMonitorTasks.HcalDataIntegrityTask_cfi    import *

from DQM.HcalMonitorTasks.HcalDetDiagLaserMonitor_cfi  import *
from DQM.HcalMonitorTasks.HcalDetDiagPedestalMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagLEDMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagNoiseMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagTimingMonitor_cfi import*

# Turn on online switches where appropriate
hcalRawDataMonitor.online         = True
hcalRecHitMonitor.online          = True
# The following tasks are not yet run online
hcalDetDiagPedestalMonitor.online = True
hcalDetDiagLaserMonitor.online    = True
hcalDetDiagNoiseMonitor.online    = True
hcalDetDiagTimingMonitor.online    = True

# Set subdetector directory to "HcalCalib/"
hcalRawDataMonitor.subSystemFolder         = "HcalCalib/"
hcalRecHitMonitor.subSystemFolder          = "HcalCalib/"
# The following tasks are not yet run online
hcalDetDiagPedestalMonitor.subSystemFolder = "HcalCalib/"
hcalDetDiagLaserMonitor.subSystemFolder    = "HcalCalib/"
hcalDetDiagNoiseMonitor.subSystemFolder    = "HcalCalib/"
hcalDetDiagTimingMonitor.subSystemFolder    = "HcalCalib/"

# Set calibration types of each monitor
hcalRawDataMonitor.AllowedCalibTypes         =  [1,2,3,4,5,6]   # check raw data quality everywhere
hcalRecHitMonitor.AllowedCalibTypes          =  [1]  # only look at pedestal
hcalDetDiagNoiseMonitor.AllowedCalibTypes    =  [0,1,2,3,4,5,6,7]
hcalDetDiagTimingMonitor.AllowedCalibTypes   =  [0,1,2,3,4,5,6,7]
# Laser & Pedestal monitors still do their own calibration type checking.
hcalDetDiagPedestalMonitor.AllowedCalibTypes =  [0,1,2,3,4,5,6,7]
hcalDetDiagLaserMonitor.AllowedCalibTypes    =  [0,1,2,3,4,5,6,7]

# Don't skip out of order LS events
hcalRawDataMonitor.skipOutOfOrderLS         =  False
hcalRecHitMonitor.skipOutOfOrderLS          =  False
hcalDetDiagPedestalMonitor.skipOutOfOrderLS =  False
hcalDetDiagLaserMonitor.skipOutOfOrderLS    =  False

# Make diagnostics where appropriate
hcalRecHitMonitor.makeDiagnostics     = True

# Specify directories where reference/outputs/etc should be directed in the Integration file!

