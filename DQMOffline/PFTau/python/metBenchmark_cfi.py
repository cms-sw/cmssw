import FWCore.ParameterSet.Config as cms


metBenchmark = cms.EDAnalyzer("METBenchmarkAnalyzer",
                              InputCollection = cms.InputTag('pfMet'),
                              mode = cms.int32( 1 ),
                              ptMin = cms.double( 20 ),
                              ptMax = cms.double( 999999 ),
                              phiMin = cms.double(-10),
                              phiMax = cms.double(10),
                              BenchmarkLabel = cms.string('pfMet')
                              )
