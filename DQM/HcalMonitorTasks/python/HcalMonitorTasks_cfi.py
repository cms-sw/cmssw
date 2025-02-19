import FWCore.ParameterSet.Config as cms

from DQM.HcalMonitorTasks.HcalDigiMonitor_cfi            import *
from DQM.HcalMonitorTasks.HcalHotCellMonitor_cfi         import *
from DQM.HcalMonitorTasks.HcalDeadCellMonitor_cfi        import *
from DQM.HcalMonitorTasks.HcalRecHitMonitor_cfi          import *
from DQM.HcalMonitorTasks.HcalNZSMonitor_cfi             import *
from DQM.HcalMonitorTasks.HcalBeamMonitor_cfi            import *
from DQM.HcalMonitorTasks.HcalRawDataMonitor_cfi         import *
from DQM.HcalMonitorTasks.HcalTrigPrimMonitor_cfi        import *

from DQM.HcalMonitorTasks.HcalDataIntegrityTask_cfi      import *

from DQM.HcalMonitorTasks.HcalDetDiagLaserMonitor_cfi    import *
from DQM.HcalMonitorTasks.HcalDetDiagPedestalMonitor_cfi import*
from DQM.HcalMonitorTasks.HcalDetDiagLEDMonitor_cfi      import*
from DQM.HcalMonitorTasks.HcalDetDiagNoiseMonitor_cfi    import*
from DQM.HcalMonitorTasks.HcalDetDiagTimingMonitor_cfi   import*

from DQM.HcalMonitorTasks.HcalLSbyLSMonitor_cfi          import*
from DQM.HcalMonitorTasks.HcalCoarsePedestalMonitor_cfi  import *
from DQM.HcalMonitorTasks.HcalNoiseMonitor_cfi  import *

hcalMonitorTasksTestSequence=cms.Sequence(hcalDigiMonitor
                                          *hcalHotCellMonitor
                                          *hcalDeadCellMonitor
                                          *hcalRecHitMonitor
                                          *hcalBeamMonitor
                                          *hcalRawDataMonitor
                                          *hcalTrigPrimMonitor
                                          *hcalNZSMonitor
                                          *hcalLSbyLSMonitor
                                          )

hcalMonitorTasksOnlineSequence = cms.Sequence(hcalDigiMonitor
                                              *hcalHotCellMonitor
                                              *hcalDeadCellMonitor
                                              *hcalRecHitMonitor
                                              *hcalBeamMonitor
                                              *hcalRawDataMonitor
                                              *hcalTrigPrimMonitor
                                              *hcalCoarsePedestalMonitor
                                              #*hcalDetDiagPedestalMonitor
                                              #*hcalDetDiagLaserMonitor
                                              #*hcalDetDiagLEDMonitor
                                              *hcalDetDiagNoiseMonitor
                                              *hcalDetDiagTimingMonitor
                                              *hcalNZSMonitor
                                              )

hcalMonitorTasksOfflineSequence = cms.Sequence(hcalDigiMonitor
                                               *hcalHotCellMonitor
                                               *hcalDeadCellMonitor
                                               *hcalRecHitMonitor
                                               *hcalBeamMonitor
                                               *hcalRawDataMonitor
                                               *hcalDetDiagNoiseMonitor
                                               *hcalLSbyLSMonitor
                                               *hcalNoiseMonitor
                                               )


hcalMonitorTasksCalibrationSequence = cms.Sequence(hcalRecHitMonitor
                                                   *hcalRawDataMonitor
                                                   *hcalDetDiagPedestalMonitor
                                                   *hcalDetDiagLaserMonitor
                                                   #*hcalDetDiagLEDMonitor
                                                   *hcalDetDiagNoiseMonitor
                                                   *hcalDetDiagTimingMonitor
                                                   )


def SetTaskParams(process,param, value):
    # Tries to set all task parameter 'param' to the value 'value'
    newval=value
    isstring=False
    if (newval<>True and newval<>False):
        try:
            newval=string.atoi(newval)
        except:
            try:
                newval=string.atof(newval)
            except:
                isstring=True

    tasks=[hcalDigiMonitor,hcalRecHitMonitor,hcalHotCellMonitor,hcalDeadCellMonitor,
           hcalRawDataMonitor, hcalBeamMonitor, hcalTrigPrimMonitor, hcalNZSMonitor,
           hcalDataIntegrityMonitor, hcalDetDiagLaserMonitor, hcalDetDiagLEDMonitor,
           hcalDetDiagNoiseMonitor, hcalDetDiagPedestalMonitor, hcalCoarsePedestalMonitor,
           hcalDetDiagTimingMonitor, hcalLSbyLSMonitor]

    for i in tasks:
        if isstring==False:
            cmd="process.%s.%s=%s"%(i,param,value)
        else:
            cmd="process.%s.%s='%s'"%(i,param,value)
        try:
            exec(cmd)
        except SyntaxError:
            print "Could not execute command '%s'"%cmd
