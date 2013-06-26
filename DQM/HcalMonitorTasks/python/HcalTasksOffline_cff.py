import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorTasks.HcalDigiMonitor_cfi          import *
from DQM.HcalMonitorTasks.HcalHotCellMonitor_cfi       import *
from DQM.HcalMonitorTasks.HcalDeadCellMonitor_cfi      import *
from DQM.HcalMonitorTasks.HcalRecHitMonitor_cfi        import *
from DQM.HcalMonitorTasks.HcalNZSMonitor_cfi           import *
from DQM.HcalMonitorTasks.HcalNoiseMonitor_cfi         import *
from DQM.HcalMonitorTasks.HcalBeamMonitor_cfi          import *
from DQM.HcalMonitorTasks.HcalRawDataMonitor_cfi       import *
from DQM.HcalMonitorTasks.HcalTrigPrimMonitor_cfi      import *
from DQM.HcalMonitorTasks.HcalDataIntegrityTask_cfi    import *

from DQM.HcalMonitorTasks.HcalDetDiagLaserMonitor_cfi  import *
from DQM.HcalMonitorTasks.HcalDetDiagPedestalMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagLEDMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagNoiseMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalLSbyLSMonitor_cfi import*

# Turn on online switches where appropriate
hcalRawDataMonitor.online         = False
hcalDigiMonitor.online            = False
hcalRecHitMonitor.online          = False
hcalHotCellMonitor.online         = False
hcalDeadCellMonitor.online        = False
hcalBeamMonitor.online            = False
hcalTrigPrimMonitor.online        = False
hcalNZSMonitor.online             = False
hcalNoiseMonitor.online           = False
hcalLSbyLSMonitor.online          = False
# The following tasks are not yet run online
hcalDetDiagLEDMonitor.online      = False
hcalDetDiagPedestalMonitor.online = False
hcalDetDiagLaserMonitor.online    = False
hcalDetDiagNoiseMonitor.online    = False

# Offline tasks should look at all events, I think
hcalRawDataMonitor.AllowedCalibTypes         =  [0,1,2,3,4,5,6,7]
hcalDigiMonitor.AllowedCalibTypes            =  [0,1,2,3,4,5,6,7]
hcalRecHitMonitor.AllowedCalibTypes          =  [0,1,2,3,4,5,6,7]
hcalHotCellMonitor.AllowedCalibTypes         =  [0,1,2,3,4,5,6,7]
hcalDeadCellMonitor.AllowedCalibTypes        =  [0,1,2,3,4,5,6,7]
hcalBeamMonitor.AllowedCalibTypes            =  [0,1,2,3,4,5,6,7]
hcalTrigPrimMonitor.AllowedCalibTypes        =  [0,1,2,3,4,5,6,7]
hcalNZSMonitor.AllowedCalibTypes             =  [0,1,2,3,4,5,6,7]
hcalNoiseMonitor.AllowedCalibTypes           =  [0,1,2,3,4,5,6,7]
hcalLSbyLSMonitor.AllowedCalibTypes          =  [0,1,2,3,4,5,6,7]

# No need to skip out of order LS in offline, I think
hcalRawDataMonitor.skipOutOfOrderLS         =  False
hcalDigiMonitor.skipOutOfOrderLS            =  False
hcalRecHitMonitor.skipOutOfOrderLS          =  False
hcalHotCellMonitor.skipOutOfOrderLS         =  False
hcalDeadCellMonitor.skipOutOfOrderLS        =  False
hcalBeamMonitor.skipOutOfOrderLS            =  False
hcalTrigPrimMonitor.skipOutOfOrderLS        =  False
hcalNZSMonitor.skipOutOfOrderLS             =  False
hcalNoiseMonitor.skipOutOfOrderLS           =  False
hcalLSbyLSMonitor.skipOutOfOrderLS          =  False

# Make diagnostics where appropriate
hcalDeadCellMonitor.makeDiagnostics   = True
hcalRecHitMonitor.makeDiagnostics     = True
hcalDigiMonitor.makeDiagnostics       = True

# Require at least N events for the dead cell monitor to process at end of lumi block?  Do we want this functionality in offline, or do we want to rely only on the never-present test?
hcalDeadCellMonitor.minDeadEventCount = 1000

# Require at least 200 events in a lumi block when looking for persistent hot cells
hcalHotCellMonitor.minEvents = 200

hcalLSbyLSMonitor.minEvents = 1000
