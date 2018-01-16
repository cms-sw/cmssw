import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
metBenchmark = DQMEDAnalyzer('METBenchmarkAnalyzer',
                              InputCollection = cms.InputTag('pfMet'),
                              mode = cms.int32( 1 ),
                              ptMin = cms.double( 0.0 ),
                              ptMax = cms.double( 999999. ),
                              phiMin = cms.double(-10.),
                              phiMax = cms.double(10.),
                              BenchmarkLabel = cms.string('pfMet')
                              )

matchMetBenchmark = DQMEDAnalyzer('MatchMETBenchmarkAnalyzer',
                                    InputCollection = cms.InputTag('pfMet'),
                                    MatchCollection = cms.InputTag('genMetTrue'),
                                    dRMax = cms.double( 999. ),
#                                    ptMin = cms.double( 0.0 ),
#                                    ptMax = cms.double( 999999 ),
#                                    etaMin = cms.double(-10),
#                                    etaMax = cms.double(10),
#                                    phiMin = cms.double(-10),
#                                    phiMax = cms.double(10),
                                    mode = cms.int32( 1 ),
                                    BenchmarkLabel = cms.string('MatchMETManager')
                                    )


