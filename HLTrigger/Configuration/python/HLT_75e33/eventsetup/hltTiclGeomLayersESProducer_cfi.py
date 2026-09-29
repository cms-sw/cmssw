import FWCore.ParameterSet.Config as cms

hltTiclGeomLayersESProducer = cms.ESProducer('TICLGeomLayersESProducer@alpaka',
    appendToDataLabel = cms.string('')
)
