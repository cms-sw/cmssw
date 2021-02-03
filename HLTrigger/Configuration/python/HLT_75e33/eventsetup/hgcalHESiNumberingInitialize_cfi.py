import FWCore.ParameterSet.Config as cms

hgcalHESiNumberingInitialize = cms.ESProducer("HGCalNumberingInitialization",
    Name = cms.untracked.string('HGCalHESiliconSensitive')
)
