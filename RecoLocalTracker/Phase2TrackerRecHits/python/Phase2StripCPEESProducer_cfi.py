import FWCore.ParameterSet.Config as cms

phase2StripCPEESProducer = cms.ESProducer("Phase2StripCPEESProducer",
  ComponentType = cms.string('Phase2StripCPETrivial'),
  parameters    = cms.PSet()
)
