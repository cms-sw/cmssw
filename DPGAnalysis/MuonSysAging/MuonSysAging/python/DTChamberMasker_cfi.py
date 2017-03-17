import FWCore.ParameterSet.Config as cms

DTChamberMasker = cms.EDProducer('DTChamberMasker',
                                 digiTag = cms.InputTag('simMuonDTDigis'),
                                 triggerPrimPhTag = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                 triggerPrimThTag = cms.InputTag('simDtTriggerPrimitiveDigis'),
                                 doTriggerFromDDU = cms.bool(True),
                                 maskedChRegEx = cms.vstring("WH0_ST1_SEC1",
                                                             "WH0_ST1_SEC3",
                                                             "WH0_ST1_SEC5",
                                                             "WH0_ST1_SEC7",
                                                             "WH0_ST1_SEC9",
                                                             "WH0_ST1_SEC11")
)
