import FWCore.ParameterSet.Config as cms

hltLSTGeometry = cms.ESProducer('LSTGeometryESProducer',
    ptCut = cms.double(0.8),
)
