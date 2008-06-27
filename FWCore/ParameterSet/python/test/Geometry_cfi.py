import FWCore.ParameterSet.Config as cms

geometry = cms.ESSource("Geometry", bField=cms.double(4.0), preshower = cms.bool(True))

