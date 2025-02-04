import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingEGammaCollectionMonitoring = DQMEDAnalyzer('ScoutingEGammaCollectionMonitoring',

    OutputInternalPath = cms.string('/HLT/ScoutingOffline/EGamma/Collection'),
    TriggerResultTag   = cms.InputTag("TriggerResults", "", "HLT"),
    ElectronCollection = cms.InputTag('slimmedElectrons'),
    ScoutingElectronCollection = cms.InputTag("hltScoutingEgammaPacker"),
    eleIdMapTight = cms.InputTag('egmGsfElectronIDsForScoutingDQM:cutBasedElectronID-RunIIIWinter22-V1-loose')
)


scoutingMonitoringEGM = cms.Sequence(ScoutingEGammaCollectionMonitoring)
