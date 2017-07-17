import FWCore.ParameterSet.Config as cms

evtplaneFilter = cms.EDFilter("EvtPlaneFilter",
                              EPlabel = cms.InputTag("hiEvtPlane"),
                              Vnlow = cms.double(0.0),
                              Vnhigh = cms.double(1.),
                              EPlvl = cms.int32(0),
                              EPidx = cms.int32(8) # HF2
        )
