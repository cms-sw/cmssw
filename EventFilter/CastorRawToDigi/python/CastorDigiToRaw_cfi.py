import FWCore.ParameterSet.Config as cms

castorRawData = cms.EDProducer("CastorDigiToRaw",
    CASTOR = cms.InputTag("simCastorDigis"),
    CastorCtdc = cms.bool(False)
)
# foo bar baz
# GzN4PK8LlaUv8
