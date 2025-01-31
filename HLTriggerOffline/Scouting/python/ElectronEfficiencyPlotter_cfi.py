import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester



ScoutingElectronEfficiencySummary = DQMEDHarvester("ElectronEfficiencyPlotter",
                                                   folder = cms.string('ScoutingMonitoring'),
                                                   srcFolder = cms.string('ScoutingMonitoring'),
                                                   ptMin = cms.double(0),
                                                   ptMax = cms.double(100),
                                                   ptBin = cms.int32(5),
                                                   sctElectronID = cms.string("scoutingID")
                                                  )


scoutingElectronEfficiencyHarvest = cms.Sequence(ScoutingElectronEfficiencySummary)
