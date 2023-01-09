import FWCore.ParameterSet.Config as cms

hgcalTBEENumberingInitialize = cms.ESProducer("HGCalTBNumberingInitialization",
                                              name = cms.untracked.string("HGCalEESensitive")
)

hgcalTBHESiNumberingInitialize = cms.ESProducer("HGCalTBNumberingInitialization",
                                                name = cms.untracked.string("HGCalHESiliconSensitive")
)
