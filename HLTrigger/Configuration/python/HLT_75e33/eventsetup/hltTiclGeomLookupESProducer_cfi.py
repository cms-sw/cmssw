import FWCore.ParameterSet.Config as cms

hltTiclGeomLookupESProducer = cms.ESProducer('TICLGeomLookupESProducer@alpaka',
    src = cms.ESInputTag('', ''),
    appendToDataLabel = cms.string('')
)
