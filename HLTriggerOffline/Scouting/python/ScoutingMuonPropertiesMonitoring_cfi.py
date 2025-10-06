import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingMuonPropertiesMonitor = DQMEDAnalyzer('ScoutingMuonPropertiesAnalyzer',
                                              OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/Properties'), #Output of the root file
                                              triggerResults = cms.InputTag("TriggerResults", "", "HLT"),
                                              muonsNoVtx = cms.InputTag("hltScoutingMuonPackerNoVtx", "", "HLT"),
                                              muonsVtx = cms.InputTag("hltScoutingMuonPackerVtx", "", "HLT"),
                                              PV = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx", "HLT"),
                                              SVNoVtx = cms.InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx", "HLT"),
                                              SVVtx = cms.InputTag("hltScoutingMuonPackerVtx", "displacedVtx", "HLT"))
    
