import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingMuonPropertiesMonitor = DQMEDAnalyzer('ScoutingMuonPropertiesAnalyzer',
                                              OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/Properties'), #Output of the root file
                                              fillAllHistograms = cms.bool(False),
                                              triggerResults = cms.InputTag("TriggerResults", "", "HLT"),
                                              muonsNoVtx = cms.InputTag("hltScoutingMuonPackerNoVtx"),
                                              muonsVtx = cms.InputTag("hltScoutingMuonPackerVtx"),
                                              PV = cms.InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"),
                                              SVNoVtx = cms.InputTag("hltScoutingMuonPackerNoVtx", "displacedVtx"),
                                              SVVtx = cms.InputTag("hltScoutingMuonPackerVtx", "displacedVtx"))    
