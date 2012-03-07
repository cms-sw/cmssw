import FWCore.ParameterSet.Config as cms
hltL1GtObjectMap = cms.EDProducer("ConvertObjectMapRecord",
                                  L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" )
)
