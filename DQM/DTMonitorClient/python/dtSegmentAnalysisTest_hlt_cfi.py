import FWCore.ParameterSet.Config as cms

dtSegmentTestkHLT = cms.EDAnalyzer("DTSegmentAnalysisTest",
                                   detailedAnalysis = cms.untracked.bool(False),
                                   #Names of the quality tests: they must match those specified in "qtList"
                                   chi2TestName = cms.untracked.string('chi2InRange'),
                                   segmRecHitTestName = cms.untracked.string('segmRecHitInRange'),
                                   #Permetted value of chi2 segment quality
                                   chi2Threshold = cms.untracked.double(5.0),
                                   normalizeHistoPlots = cms.untracked.bool(False),
                                   # top folder for the histograms in DQMStore
                                   topHistoFolder = cms.untracked.string("HLT/HLTMonMuon/DT-Segments"),
                                   # hlt DQM mode
                                   hltDQMMode = cms.untracked.bool(True)
                                   )


