import FWCore.ParameterSet.Config as cms

hfnoseParametersInitialize = cms.ESProducer("HGCalParametersESModule",
                                            Name  = cms.untracked.string("HGCalHFNoseSensitive"),
                                            NameW = cms.untracked.string("HFNoseWafer"),
                                            NameC = cms.untracked.string("HFNoseCell"),
                                            NameT = cms.untracked.string("HFNose")
)
