import FWCore.ParameterSet.Config as cms

conditionDumperInEdm = cms.EDProducer("ConditionDumperInEdm",
                                      gtEvmDigisLabel = cms.InputTag("gtEvmDigis")
                                      )

