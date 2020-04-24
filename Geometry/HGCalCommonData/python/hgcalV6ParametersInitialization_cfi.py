import FWCore.ParameterSet.Config as cms

hgcalEEParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                             Name = cms.untracked.string("HGCalEESensitive"),
                                             NameW = cms.untracked.string("HGCalEEWafer"),
                                             NameC = cms.untracked.string("HGCalEECell")
)

hgcalHESiParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                               Name = cms.untracked.string("HGCalHESiliconSensitive"),
                                               NameW = cms.untracked.string("HGCalHEWafer"),
                                               NameC = cms.untracked.string("HGCalHECell")
)

