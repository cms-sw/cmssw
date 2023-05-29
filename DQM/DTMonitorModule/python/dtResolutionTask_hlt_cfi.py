import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtResolutionTaskHLT = DQMEDAnalyzer('DTResolutionAnalysisTask',
                                     # labels of 4D hits
                                     recHits4DLabel = cms.untracked.string('hltDt4DSegments'),
                                     # interval of lumi block after which we reset the histos
                                     ResetCycle = cms.untracked.int32(10000),
                                     # top folder for the histograms in DQMStore
                                     topHistoFolder = cms.untracked.string('HLT/HLTMonMuon/DT-Segments')
                                     )


