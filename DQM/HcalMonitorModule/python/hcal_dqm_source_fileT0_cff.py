import FWCore.ParameterSet.Config as cms

#from DQMServices.Components.DQMEnvironment_cfi import *

dqmInfoHcal = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Hcal')
)
from DQM.HcalMonitorModule.HcalMonitorModule_cfi import *
hcalOfflineDQMSource = cms.Sequence(hcalMonitor + dqmInfoHcal)

# Loosen HF hot cell thresholds when using cosmic reconstruction
hcalMonitor.HotCellMonitor_HF_energyThreshold = 20
hcalMonitor.HotCellMonitor_HF_persistentThreshold = 10
hcalMonitor.ReferencePedestalMonitor        = False
hcalMonitor.DataFormatMonitor      = True
hcalMonitor.DigiMonitor            = True
hcalMonitor.RecHitMonitor          = True
hcalMonitor.BeamMonitor            = True
# Disable until emulated trigger digis are included in sequence
hcalMonitor.TrigPrimMonitor        = False
hcalMonitor.DetDiagNoiseMonitor    = True
hcalMonitor.DetDiagTimingMonitor   = True
hcalMonitor.DeadCellMonitor        = True
hcalMonitor.HotCellMonitor         = True
hcalMonitor.LEDMonitor             = False
hcalMonitor.MTCCMonitor            = False
hcalMonitor.CaloTowerMonitor       = False
hcalMonitor.HcalAnalysis           = False

