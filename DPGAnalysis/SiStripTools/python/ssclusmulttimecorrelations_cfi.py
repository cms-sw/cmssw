import FWCore.ParameterSet.Config as cms

ssclusmulttimecorrelations = cms.EDAnalyzer('MultiplicityTimeCorrelations',
                          wantedSubDets = cms.untracked.VPSet(    
                            cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712/64), phasePartition = cms.untracked.string("All")),
                            cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/64), phasePartition = cms.untracked.string("TI")),
                            cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID"), binMax = cms.int32( 565248/64), phasePartition = cms.untracked.string("TI")),
                            cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB"), binMax = cms.int32(3303936/64), phasePartition = cms.untracked.string("TO")),
                            cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"), binMax = cms.int32(3866624/64), phasePartition = cms.untracked.string("TP"))
                         ),
                                         hitName = cms.untracked.string("cluster"),
                                         historyProduct = cms.InputTag("froml1abcHEs"),
                                         apvPhaseCollection = cms.InputTag("APVPhases"),
                                         multiplicityMap = cms.InputTag("ssclustermultprod"),
                                         scaleFactors = cms.untracked.vint32(10),
                                         numberOfBins = cms.untracked.int32(500),
                                         corrNbins = cms.untracked.int32(1000),
                                         lowedgeOrbit = cms.untracked.int32(-1),
                                         highedgeOrbit = cms.untracked.int32(-1),
                                         minDBX = cms.untracked.int32(-1),
                                         minTripletDBX = cms.untracked.int32(-1),
                                         dbxBins = cms.untracked.vint32()
                                      )

