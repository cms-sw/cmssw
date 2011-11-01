import FWCore.ParameterSet.Config as cms

effPlotter = cms.EDAnalyzer("EfficiencyPlotter",
                            phiMin = cms.double(-3.2),
                            etaMin = cms.double(-2.5),
                            ptMin  = cms.double(0.),
                            etaBin = cms.int32(100),
                            ptBin = cms.int32(100),
                            phiBin = cms.int32(100),
                            etaMax = cms.double(2.5),
                            phiMax = cms.double(3.2),
                            ptMax  = cms.double(200.)
                            )



