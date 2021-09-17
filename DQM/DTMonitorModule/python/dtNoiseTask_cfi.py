import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtNoiseMonitor = DQMEDAnalyzer('DTNoiseTask',
                                # the label to retrieve the DT digis
                                dtDigiLabel = cms.InputTag('dtunpacker'),
                                # switch for time box booking
                                doTbHistos = cms.untracked.bool(False),
                                # the name of the 4D rec hits collection
                                recHits4DLabel = cms.string('dt4DSegments'),
                                # switch for segment veto
                                doSegmentVeto = cms.untracked.bool(False),
                                # safe margin (ns) between ttrig and beginning of counting area
                                safeMargin = cms.untracked.double(100.)
                                )
