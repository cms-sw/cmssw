import FWCore.ParameterSet.Config as cms

ecalFEDWithCRCErrorProducer = cms.EDProducer("EcalFEDWithCRCErrorProducer",
    InputLabel = cms.InputTag("source")

)
