import FWCore.ParameterSet.Config as cms

#from DQMServices.Components.DQMEnvironment_cfi import *

dqmInfoHcal = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Hcal')
)
from DQM.HcalMonitorModule.HcalMonitorModule_cfi import *
hcalOfflineDQMSource = cms.Sequence(hcalMonitor + dqmInfoHcal)
hcalMonitor.DigiOccThresh          = -999999999
hcalMonitor.PedestalsPerChannel    = False
hcalMonitor.checkNevents           = 500

hcalMonitor.PedestalsInFC          = True
hcalMonitor.DataFormatMonitor      = True
hcalMonitor.DataIntegrityTask      = True
hcalMonitor.DigiMonitor            = True
hcalMonitor.RecHitMonitor          = False
hcalMonitor.BeamMonitor            = False
hcalMonitor.TrigPrimMonitor        = False
hcalMonitor.DeadCellMonitor        = True
hcalMonitor.HotCellMonitor         = True
hcalMonitor.PedestalMonitor        = False
hcalMonitor.LEDMonitor             = False
hcalMonitor.MTCCMonitor            = False
hcalMonitor.CaloTowerMonitor       = False
hcalMonitor.HcalAnalysis           = False

