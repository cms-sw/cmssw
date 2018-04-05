import FWCore.ParameterSet.Config as cms

fastTimeParametersInitialize = cms.ESProducer("FastTimeParametersESModule",
                                              Names = cms.untracked.vstring("FastTimeBarrel","SFBX"),
                                              Types = cms.untracked.vint32(1,2),
)
