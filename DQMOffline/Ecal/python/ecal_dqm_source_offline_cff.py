import FWCore.ParameterSet.Config as cms

from DQM.EcalMonitorTasks.EcalMonitorTask_cfi import *
from DQM.EcalMonitorTasks.EcalFEDMonitor_cfi import *
from DQMOffline.Ecal.ESRecoSummary_cfi import *
from DQMOffline.Ecal.EcalZmassTask_cfi import *
from DQMOffline.Ecal.EcalPileUpDepMonitor_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoEcal = DQMEDAnalyzer('DQMEventInfo',
    subSystemFolder = cms.untracked.string('Ecal')
)

## standard
ecal_dqm_source_offline = cms.Sequence(
    dqmInfoEcal +
    ecalMonitorTask +
    ecalFEDMonitor +
    ecalPreshowerRecoSummary +
    ecalzmasstask +
    ecalPileUpDepMonitor
)

ecalMonitorTask.workerParameters.TrigPrimTask.params.runOnEmul = False
