import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester



#ScoutingElectronEfficiencySummary = DQMEDHarvester("ElectronEfficiencyPlotter",
#                                                   folder = cms.string('ScoutingMonitoring/Efficiency'),
#                                                   srcFolder = cms.string('ScoutingMonitoring'),
#                                                   ptMin = cms.double(0),
#                                                   ptMax = cms.double(200),
#                                                   ptBin = cms.int32(20),
#                                                   sctElectronID = cms.string("scoutingID")
#                                                  )

ScoutingElectronTagProbeAnalysis = DQMEDAnalyzer('ScoutingElectronTagProbeAnalyzer',

    OutputInternalPath = cms.string('/HLT/ScoutingOffline/EGamma/TnP/Tag_ScoutingElectron'),
    TriggerResultTag   = cms.InputTag("TriggerResults", "", "HLT"),
    FilterToMatch      = cms.vstring("hltPreDSTHLTMuonRun3PFScoutingPixelTracking",
                                     "hltDoubleEG16EG12CaloIdLHEFilter",
                                     "hltSingleEG30CaloIdLHEFilter",
                                     "hltPreDSTRun3JetHTPFScoutingPixelTracking"),
    TriggerObjects     = cms.InputTag("slimmedPatTrigger"),
    ElectronCollection = cms.InputTag('slimmedElectrons'),
    ScoutingElectronCollection = cms.InputTag('hltScoutingEgammaPacker')

)

scoutingMonitoringTagProbe = cms.Sequence(ScoutingElectronTagProbeAnalysis) # * ScoutingElectronEfficiencySummary)
