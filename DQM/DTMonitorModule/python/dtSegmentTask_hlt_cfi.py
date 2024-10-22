import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtSegmentTaskHLT = DQMEDAnalyzer('DTSegmentAnalysisTask',
                                  # switch for verbosity
                                  debug = cms.untracked.bool(False),
                                  # label of 4D segments
                                  recHits4DLabel = cms.untracked.string('hltDt4DSegments'),
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
                                  topHistoFolder = cms.untracked.string('HLT/HLTMonMuon/DT-Segments'),
                                  # hlt DQM mode
                                  hltDQMMode = cms.untracked.bool(True)
                                  )


