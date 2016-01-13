import FWCore.ParameterSet.Config as cms

hgcalEEParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                             Name  = cms.untracked.string("HGCalEESensitive"),
                                             NameW = cms.untracked.string("HGCalWafer"),
                                             NameC = cms.untracked.string("HGCalCell")
)

hgcalHESiParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                               Name  = cms.untracked.string("HGCalHESiliconSensitive"),
                                               NameW = cms.untracked.string("HGCalWafer"),
                                               NameC = cms.untracked.string("HGCalCell")
)

hgcalHEScParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                               Name = cms.untracked.string("HGCalHEScintillatorSensitive"),
                                               NameW = cms.untracked.string("HGCalWafer"),
                                               NameC = cms.untracked.string("HGCalCell")
)
