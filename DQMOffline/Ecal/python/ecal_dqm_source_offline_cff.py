import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi import *
from DQM.EcalBarrelMonitorTasks.EcalFEDMonitor_cfi import *
from DQMOffline.Ecal.ESRecoSummary_cfi import *
from DQMOffline.Ecal.EcalZmassTask_cfi import *

# modules that are referenced in DQMOffline.Configuration.DQMOfflineMC_cff
ecalBarrelRawDataTask = cms.PSet(FEDRawDataCollection = cms.untracked.string(''))
ecalEndcapRawDataTask = cms.PSet(FEDRawDataCollection = cms.untracked.string(''))
ecalBarrelSelectiveReadoutTask = cms.PSet(FEDRawDataCollection = cms.untracked.string(''))
ecalEndcapSelectiveReadoutTask = cms.PSet(FEDRawDataCollection = cms.untracked.string(''))
ecalBarrelHltTask = cms.PSet(FEDRawDataCollection = cms.untracked.string(''))
ecalEndcapHltTask = cms.PSet(FEDRawDataCollection = cms.untracked.string(''))

dqmInfoEcal = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Ecal')
)

## standard
ecal_dqm_source_offline = cms.Sequence(
    dqmInfoEcal +
    ecalMonitorTask +
    ecalFEDMonitor +
    ecalPreshowerRecoSummary +
    ecalzmasstask
)
