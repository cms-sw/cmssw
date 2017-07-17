import FWCore.ParameterSet.Config as cms

spclusmultinvestigator = cms.EDAnalyzer('MultiplicityInvestigator',
                                        vertexCollection = cms.InputTag(""),
                                        wantInvestHist = cms.bool(True),
                                        wantVtxCorrHist = cms.bool(False),
                                        wantLumiCorrHist = cms.bool(False),
                                        wantPileupCorrHist = cms.bool(False),
                                        wantVtxPosCorrHist = cms.bool(False),
                                        wantedSubDets = cms.untracked.VPSet(    
                               cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel"), binMax = cms.int32(200000)),
                               cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
                               cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIX"), binMax = cms.int32(100000))
                              ),
                                         hitName = cms.untracked.string("cluster"),
                                         multiplicityMap = cms.InputTag("spclustermultprod"),
                                         numberOfBins = cms.untracked.int32(500),
                                         maxLSBeforeRebin = cms.untracked.uint32(100),
                                         startingLSFraction = cms.untracked.uint32(4),
                                         scaleFactor = cms.untracked.int32(100),
                                      )

