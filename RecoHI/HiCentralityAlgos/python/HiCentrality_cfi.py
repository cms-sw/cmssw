import FWCore.ParameterSet.Config as cms

hiCentrality = cms.EDProducer("CentralityProducer",
                              recoLevel = cms.untracked.bool(True),
                              srcHF = cms.InputTag("hfreco") # for 3_1_X
                              )


