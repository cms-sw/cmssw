import FWCore.ParameterSet.Config as cms

# The pixel hit errors come from the genError payload of the GlobalTag (SiPixelGenErrorDBObjectRcd), the same
# payload as hltESPPixelCPEGeneric.
hltESPPixelCPEFastParamsPhase2 = cms.ESProducer('PixelCPEFastParamsESProducerAlpakaPhase2@alpaka', 
    ComponentName = cms.string("PixelCPEFastParamsPhase2"),
    LoadTemplatesFromDB = cms.bool(True),
    UseErrorsFromTemplates = cms.bool(True),
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
