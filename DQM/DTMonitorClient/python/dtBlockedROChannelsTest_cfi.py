import FWCore.ParameterSet.Config as cms

blockedROChannelTest = cms.EDAnalyzer("DTBlockedROChannelsTest",
                                      diagnosticPrescale = cms.untracked.int32(1)
                                      )


