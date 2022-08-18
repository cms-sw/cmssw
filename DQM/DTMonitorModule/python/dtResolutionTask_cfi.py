import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtResolutionAnalysisMonitor = DQMEDAnalyzer('DTResolutionAnalysisTask',
                                             # labels of 4D hits
                                             recHits4DLabel = cms.untracked.string('dt4DSegments'),
                                             # interval of lumi block after which we reset the histos
                                             ResetCycle = cms.untracked.int32(10000),
                                             # cut on the hits of segments considered for resolution
                                             phiHitsCut = cms.untracked.uint32(6),
                                             zHitsCut = cms.untracked.uint32(3),
                                             # top folder for the histograms in DQMStore
                                             topHistoFolder = cms.untracked.string('DT/02-Segments')
                                             )


