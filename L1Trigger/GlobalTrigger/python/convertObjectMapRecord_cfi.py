import FWCore.ParameterSet.Config as cms
convertObjectMapRecord= cms.EDProducer("ConvertObjectMapRecord",
                                  L1GtObjectMapTag=cms.InputTag("hltL1GtObjectMap")
                                  )

