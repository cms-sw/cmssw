
import FWCore.ParameterSet.Config as cms

l1GtExtCondLegacyToStage2 = cms.EDProducer("L1TExtCondLegacyToStage2",
                                           bxFirst = cms.int32(-2),
                                           bxLast = cms.int32(2),
                                           LegacyGtReadoutRecord = cms.InputTag("unpackLegacyGtDigis")
)

