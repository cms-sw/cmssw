import FWCore.ParameterSet.Config as cms

blockedROChannelTest = cms.EDAnalyzer("DTBlockedROChannelsTest",
                                      offlineMode = cms.untracked.bool(False),
                                      diagnosticPrescale = cms.untracked.int32(1)
                                      )


