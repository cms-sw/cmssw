import FWCore.ParameterSet.Config as cms

spclusmulttimecorrelations = cms.EDAnalyzer('MultiplicityTimeCorrelations',
                              wantedSubDets = cms.untracked.VPSet(    
                               cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel"), binMax = cms.int32(200000)),
                               cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
                               cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("FPIX"), binMax = cms.int32(100000))
                              ),
                                         hitName = cms.untracked.string("cluster"),
                                         historyProduct = cms.InputTag("froml1abcHEs"),
                                         apvPhaseCollection = cms.InputTag("APVPhases"),
                                         multiplicityMap = cms.InputTag("spclustermultprod"),
                                         scaleFactors = cms.untracked.vint32(100),
                                         corrNbins = cms.untracked.int32(1000),
                                         numberOfBins = cms.untracked.int32(200),
                                         lowedgeOrbit = cms.untracked.int32(-1),
                                         highedgeOrbit = cms.untracked.int32(-1),
                                         minDBX = cms.untracked.int32(-1),
                                         minTripletDBX = cms.untracked.int32(-1),
                                         dbxBins = cms.untracked.vint32()
                                      )

