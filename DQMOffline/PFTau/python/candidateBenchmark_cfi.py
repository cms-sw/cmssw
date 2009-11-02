import FWCore.ParameterSet.Config as cms


candidateBenchmark = cms.EDAnalyzer("CandidateBenchmarkAnalyzer",
                                    OutputFile = cms.untracked.string('benchmark.root'),
                                    InputCollection = cms.InputTag('particleFlow'),
                                    mode = cms.int32( 1 ),
                                    BenchmarkLabel = cms.string('particleFlowCandidate')
                                    )
