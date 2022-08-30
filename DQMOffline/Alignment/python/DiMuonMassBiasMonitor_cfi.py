import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DiMuonMassBiasMonitor = DQMEDAnalyzer('DiMuonMassBiasMonitor',
                                      muonTracks = cms.InputTag('ALCARECOTkAlDiMuon'),
                                      vertices = cms.InputTag('offlinePrimaryVertices'),
                                      FolderName = cms.string('DiMuonMassBiasMonitor'),
                                      decayMotherName = cms.string('Z'),
                                      distanceScaleFactor = cms.double(0.1), # for the Z->mm
                                      DiMuMassConfig = cms.PSet(
                                          name = cms.string('DiMuMass'),
                                          title = cms.string('M(#mu#mu)'),
                                          yUnits = cms.string('[GeV]'),
                                          NxBins = cms.int32(24),
                                          NyBins = cms.int32(50),
                                          ymin = cms.double(65.),
                                          ymax = cms.double(115.)
                                      ))
