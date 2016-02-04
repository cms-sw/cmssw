import FWCore.ParameterSet.Config as cms


pfCandidateManager = cms.EDAnalyzer("PFCandidateManagerAnalyzer",
                                    InputCollection = cms.InputTag('particleFlow'),
                                    MatchCollection = cms.InputTag(''),
                                    dRMax = cms.double( 0.2 ),
                                    ptMin = cms.double( 2 ),
                                    ptMax = cms.double( 999999 ),
                                    etaMin = cms.double(-10),
                                    etaMax = cms.double(10),
                                    phiMin = cms.double(-10),
                                    phiMax = cms.double(10),
                                    matchCharge = cms.bool( True ), 
                                    mode = cms.int32( 1 ),
                                    BenchmarkLabel = cms.string('particleFlowManager')
                                    )
