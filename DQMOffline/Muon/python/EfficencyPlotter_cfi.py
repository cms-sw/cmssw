import FWCore.ParameterSet.Config as cms

effPlotter = cms.EDAnalyzer("EfficiencyPlotter",
                            phiMin = cms.double(-3.2),
                            etaMin = cms.double(-2.5),
                            ptMin  = cms.double(10),
                            etaBin = cms.int32(8),
                            ptBin = cms.int32(10),
                            phiBin = cms.int32(8),
                            etaMax = cms.double(2.5),
                            phiMax = cms.double(3.2),
                            ptMax  = cms.double(100)
                            )



