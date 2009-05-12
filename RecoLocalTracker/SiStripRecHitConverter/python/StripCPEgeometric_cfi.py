import FWCore.ParameterSet.Config as cms

StripCPEgeometricESProducer = cms.ESProducer("StripCPEESProducer",
    ComponentName = cms.string('StripCPEgeometric')
)


