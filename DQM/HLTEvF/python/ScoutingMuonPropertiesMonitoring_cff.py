import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

scoutingMuonPropertiesMonitor = DQMEDAnalyzer('ScoutingMuonPropertiesAnalyzer',
                                              triggerResults = cms.InputTag("TriggerResults", "", "HLT"),
                                              muonsNoVtx = cms.InputTag("hltScoutingMuonPackerNoVtx", "", "HLT"),
                                              muonsVtx = cms.InputTag("hltScoutingMuonPackerVtx", "", "HLT"),
                                              PV = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx", "HLT"),
                                              SVNoVtx = cms.InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx", "HLT"),
                                              SVVtx = cms.InputTag("hltScoutingMuonPackerVtx", "displacedVtx", "HLT"))
    
