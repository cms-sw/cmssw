import FWCore.ParameterSet.Config as cms

phase2StripCPEGeometricESProducer = cms.ESProducer("Phase2StripCPEESProducer",
    ComponentType = cms.string('Phase2StripCPEGeometric'),
    parameters = cms.PSet(

    )
)
