import FWCore.ParameterSet.Config as cms

hgcalEEParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                             Name  = cms.untracked.string("HGCalEESensitive"),
                                             NameW = cms.untracked.string("HGCalEEWafer"),
                                             NameC = cms.untracked.string("HGCalEECell"),
                                             NameT = cms.untracked.string("HGCal")
)

hgcalHESiParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                               Name  = cms.untracked.string("HGCalHESiliconSensitive"),
                                               NameW = cms.untracked.string("HGCalHEWafer"),
                                               NameC = cms.untracked.string("HGCalHECell"),
                                               NameT = cms.untracked.string("HGCal")
)

hgcalHEScParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                               Name = cms.untracked.string("HGCalHEScintillatorSensitive"),
                                               NameW = cms.untracked.string("HGCalWafer"),
                                               NameC = cms.untracked.string("HGCalCell"),
                                               NameT = cms.untracked.string("HGCal")
)
