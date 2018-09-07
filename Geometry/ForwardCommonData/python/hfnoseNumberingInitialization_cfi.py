import FWCore.ParameterSet.Config as cms

hfnoseNumberingInitialize = cms.ESProducer("HGCalNumberingInitialization",
                                           Name = cms.untracked.string("HGCalHFNoseSensitive")
)
