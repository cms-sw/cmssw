import FWCore.ParameterSet.Config as cms

HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP", applyAlignment = cms.untracked.bool(False))
