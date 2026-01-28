import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingDileptonMonitor = DQMEDAnalyzer("ScoutingDileptonMonitor",
                                        muons     = cms.InputTag("hltScoutingMuonPacker"),
                                        electrons = cms.InputTag("hltScoutingElectronPacker"),
                                        doMuons     = cms.bool(True),
                                        doElectrons = cms.bool(True),                                            
                                        muonCut     = cms.string("pt > 5 && abs(eta) < 2.4"),
                                        electronCut = cms.string("pt > 7 && abs(eta) < 2.5"),                           
                                        massBins = cms.int32(120),
                                        massMin  = cms.double(0.0),
                                        massMax  = cms.double(200.0),                                            
                                        zMassMin = cms.double(70.0),
                                        zMassMax = cms.double(110.0),                                            
                                        barrelEta = cms.double(1.479)
                                        )
