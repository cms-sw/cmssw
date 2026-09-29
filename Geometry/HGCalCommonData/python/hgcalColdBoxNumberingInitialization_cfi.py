import FWCore.ParameterSet.Config as cms

hgcalEENumberingInitialize = cms.ESProducer("HGCalNumberingInitialization",
                                            Name = cms.untracked.string("HGCalEESensitive")
)
