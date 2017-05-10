import FWCore.ParameterSet.Config as cms

blockedROChannelTest = cms.EDProducer("DTBlockedROChannelsTest",
                                      offlineMode = cms.untracked.bool(False),
                                      diagnosticPrescale = cms.untracked.int32(1)
                                      )


