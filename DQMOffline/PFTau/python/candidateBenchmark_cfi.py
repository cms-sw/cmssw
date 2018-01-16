import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
candidateBenchmark = DQMEDAnalyzer('CandidateBenchmarkAnalyzer',
                                    InputCollection = cms.InputTag('particleFlow'),
                                    mode = cms.int32( 1 ),
                                    ptMin = cms.double( 2 ),
                                    ptMax = cms.double( 999999 ),
                                    etaMin = cms.double(-10),
                                    etaMax = cms.double(10),
                                    phiMin = cms.double(-10),
                                    phiMax = cms.double(10),
                                    BenchmarkLabel = cms.string('particleFlowCandidate')
                                    )
