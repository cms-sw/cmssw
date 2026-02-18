import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingDileptonMonitor = DQMEDAnalyzer("ScoutingDileptonMonitor",
                                        OutputInternalPath = cms.string('HLT/ScoutingOffline/DiLepton'), #Output of the root file
                                        muons     = cms.InputTag("hltScoutingMuonPackerVtx"),
                                        muonsNoVtx  = cms.InputTag("hltScoutingMuonPackerNoVtx"),
                                        electrons = cms.InputTag("hltScoutingEgammaPacker"),
                                        doMuons     = cms.bool(True),
                                        doMuonsNoVtx  = cms.bool(True),
                                        doElectrons = cms.bool(True),
                                        muonCut     = cms.string(""), #cms.string("pt > 3 && abs(eta) < 2.4"),
                                        electronCut = cms.string(""), #cms.string("pt > 3 && abs(eta) < 2.5"),
                                        massBins = cms.int32(120),
                                        massMin  = cms.double(0.0),
                                        massMax  = cms.double(200.0),
                                        zMassMin = cms.double(70.0),
                                        zMassMax = cms.double(110.0),
                                        barrelEta = cms.double(1.479),
                                        muonID = cms.bool(True),
                                        electronID = cms.bool(True))
