import FWCore.ParameterSet.Config as cms


pfCandidateManager = cms.EDAnalyzer("PFCandidateManagerAnalyzer",
                                    InputCollection = cms.InputTag('particleFlow'),
                                    MatchCollection = cms.InputTag(''),
                                    dRMax = cms.double( 0.2 ),
                                    matchCharge = cms.bool( True ), 
                                    mode = cms.int32( 1 ),
                                    BenchmarkLabel = cms.string('particleFlowMatch')
                                    )
