import FWCore.ParameterSet.Config as cms

pfBlockBasedIsolation = cms.PSet(
    ComponentName = cms.string('pfBlockBasedIsolation'),
    coneSize = cms.double(9999999999)
)