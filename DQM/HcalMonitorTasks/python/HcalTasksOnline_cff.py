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
from DQM.HcalMonitorTasks.HcalCoarsePedestalMonitor_cfi import *

from DQM.HcalMonitorTasks.HcalDetDiagLaserMonitor_cfi  import *
from DQM.HcalMonitorTasks.HcalDetDiagPedestalMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagLEDMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagNoiseMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagTimingMonitor_cfi import*


# Turn on online switches where appropriate
hcalRawDataMonitor.online         = True
hcalDigiMonitor.online            = True
hcalRecHitMonitor.online          = True
hcalHotCellMonitor.online         = True
hcalDeadCellMonitor.online        = True
hcalBeamMonitor.online            = True
hcalTrigPrimMonitor.online        = True
hcalNZSMonitor.online             = True
hcalCoarsePedestalMonitor.online  = True
# The following tasks are not yet run online
hcalDetDiagLEDMonitor.online      = True
hcalDetDiagPedestalMonitor.online = True
hcalDetDiagLaserMonitor.online    = True
hcalDetDiagNoiseMonitor.online    = True
hcalDetDiagTimingMonitor.online    = True

# Set subdetector directory to "Hcal/"
hcalRawDataMonitor.subSystemFolder         = "Hcal/"
hcalDigiMonitor.subSystemFolder            = "Hcal/"
hcalRecHitMonitor.subSystemFolder          = "Hcal/"
hcalHotCellMonitor.subSystemFolder         = "Hcal/"
hcalDeadCellMonitor.subSystemFolder        = "Hcal/"
hcalBeamMonitor.subSystemFolder            = "Hcal/"
hcalTrigPrimMonitor.subSystemFolder        = "Hcal/"
hcalNZSMonitor.subSystemFolder             = "Hcal/"
hcalCoarsePedestalMonitor.subSystemFolder  = "Hcal/"
# The following tasks are not yet run online
hcalDetDiagLEDMonitor.subSystemFolder      = "Hcal/"
hcalDetDiagPedestalMonitor.subSystemFolder = "Hcal/"
hcalDetDiagLaserMonitor.subSystemFolder    = "Hcal/"
hcalDetDiagNoiseMonitor.subSystemFolder    = "Hcal/"
hcalDetDiagTimingMonitor.subSystemFolder    = "Hcal/"

# Only look at non-calibration events, except dead cell, which can use any
hcalRawDataMonitor.AllowedCalibTypes         =  [0]
hcalDigiMonitor.AllowedCalibTypes            =  [0]
hcalRecHitMonitor.AllowedCalibTypes          =  [0]
hcalHotCellMonitor.AllowedCalibTypes         =  [0]
hcalDeadCellMonitor.AllowedCalibTypes        =  [0,1,2,3,4,5,6,7]
hcalBeamMonitor.AllowedCalibTypes            =  [0]
hcalTrigPrimMonitor.AllowedCalibTypes        =  [0]
hcalNZSMonitor.AllowedCalibTypes             =  [0]
hcalDetDiagNoiseMonitor.AllowedCalibTypes    =  [0]
hcalDetDiagTimingMonitor.AllowedCalibTypes   =  [0]
hcalCoarsePedestalMonitor.AllowedCalibTypes  =  [1]

# Skip out of order lumi sections for hot, dead, beam monitors
hcalRawDataMonitor.skipOutOfOrderLS         =  False
hcalDigiMonitor.skipOutOfOrderLS            =  False
hcalRecHitMonitor.skipOutOfOrderLS          =  False
hcalHotCellMonitor.skipOutOfOrderLS         =  True
hcalDeadCellMonitor.skipOutOfOrderLS        =  True
hcalBeamMonitor.skipOutOfOrderLS            =  True
hcalTrigPrimMonitor.skipOutOfOrderLS        =  False
hcalNZSMonitor.skipOutOfOrderLS             =  False

# Make diagnostics where appropriate
hcalDeadCellMonitor.makeDiagnostics   = True
hcalRecHitMonitor.makeDiagnostics     = True

# Require at least 2000 events for the dead cell monitor to process at end of lumi block
hcalDeadCellMonitor.minDeadEventCount = 2000

# Require at least 200 event in a lumi block when looking for persistent hot cells
hcalHotCellMonitor.minEvents = 200


# Specify directories where reference/outputs/etc should be directed in the Integration file!

