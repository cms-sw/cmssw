import FWCore.ParameterSet.Config as cms

ecalFEDErrorFilter = cms.EDFilter("EcalFEDErrorFilter",
    InputLabel = cms.InputTag("source")

)
