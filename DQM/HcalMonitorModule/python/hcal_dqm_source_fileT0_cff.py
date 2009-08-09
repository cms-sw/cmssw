import FWCore.ParameterSet.Config as cms

#from DQMServices.Components.DQMEnvironment_cfi import *

dqmInfoHcal = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Hcal')
)
from DQM.HcalMonitorModule.HcalMonitorModule_cfi import *
hcalOfflineDQMSource = cms.Sequence(hcalMonitor + dqmInfoHcal)

hcalMonitor.PedestalMonitor        = False
hcalMonitor.DataFormatMonitor      = True
hcalMonitor.DigiMonitor            = True
hcalMonitor.RecHitMonitor          = True
hcalMonitor.BeamMonitor            = True
hcalMonitor.TrigPrimMonitor        = False # disabled until we find why it crashes
hcalMonitor.DetDiagNoiseMonitor    = True
hcalMonitor.DeadCellMonitor        = True
hcalMonitor.HotCellMonitor         = True
hcalMonitor.LEDMonitor             = False
hcalMonitor.MTCCMonitor            = False
hcalMonitor.CaloTowerMonitor       = False
hcalMonitor.HcalAnalysis           = False

