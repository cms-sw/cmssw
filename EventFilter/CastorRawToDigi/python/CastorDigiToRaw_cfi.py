import FWCore.ParameterSet.Config as cms

castorRawData = cms.EDProducer("CastorDigiToRaw",
    CASTOR = cms.untracked.InputTag("simCastorDigis"),
)
