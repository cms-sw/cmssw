import FWCore.ParameterSet.Config as cms

dttfpacker = cms.EDFilter("DTTFFEDSim",
    DTTracks_Source = cms.InputTag("dttfDigis","DTTF"),
    DTDigi_Source = cms.InputTag("dtTriggerPrimitiveDigis")
)


