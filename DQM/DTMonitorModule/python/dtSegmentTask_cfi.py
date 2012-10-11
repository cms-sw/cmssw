import FWCore.ParameterSet.Config as cms

dtSegmentAnalysisMonitor = cms.EDAnalyzer("DTSegmentAnalysisTask",
                                          # switch for verbosity
                                          debug = cms.untracked.bool(False),
                                          # label of 4D segments
                                          recHits4DLabel = cms.string('dt4DSegments'),
                                          # skip segments with noisy cells (reads from DB)
                                          checkNoisyChannels = cms.untracked.bool(True),
                                          # switch off uneeded histograms
                                          detailedAnalysis = cms.untracked.bool(False),
                                          # # of bins in the time histos
                                          nTimeBins = cms.untracked.int32(100),
                                          # # of LS per bin in the time histos
                                          nLSTimeBin = cms.untracked.int32(15),
                                          # switch on/off sliding bins in time histos
                                          slideTimeBins = cms.untracked.bool(True),
                                          # top folder for the histograms in DQMStore
                                          topHistoFolder = cms.untracked.string("DT/02-Segments"),
                                          # hlt DQM mode
                                          hltDQMMode = cms.untracked.bool(False),
                                          # max phi angle of reconstructed segments 
                                          phiSegmCut = cms.untracked.double(30.),
                                          # min # hits of segment used to validate a segment in WB+-2/SecX/MB1 
                                          nhitsCut = cms.untracked.int32(11)
                                          )


