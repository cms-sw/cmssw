import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ScoutingElectronEfficiencySummary = DQMEDHarvester("ElectronEfficiencyPlotter",
                                                   folder = cms.string('/HLT/ScoutingOffline/EGamma/Efficiency'),
                                                   srcFolder = cms.string('/HLT/ScoutingOffline/EGamma/TnP/Tag_PatElectron'),
                                                   triggerSelection = cms.vstring(["DST_PFScouting_DoubleEG_v", "DST_PFScouting_SinglePhotonEB_v"]),
                                                   ptMin = cms.double(0),
                                                   ptMax = cms.double(100),
                                                   ptBin = cms.int32(5),
                                                   sctElectronID = cms.string("scoutingID")
                                                   )

scoutingElectronEfficiencyHarvest = cms.Sequence(ScoutingElectronEfficiencySummary)
