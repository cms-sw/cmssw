import FWCore.ParameterSet.Config as cms

castorRawData = cms.EDFilter("CastorDigiToRaw",
    CASTOR = cms.untracked.InputTag("castorDigis"),
)
