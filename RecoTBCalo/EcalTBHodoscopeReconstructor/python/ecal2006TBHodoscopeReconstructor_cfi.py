import FWCore.ParameterSet.Config as cms

#Hodoscope reconstruction
ecal2006TBHodoscopeReconstructor = cms.EDProducer("EcalTBHodoscopeRecInfoProducer",
    fitMethod = cms.int32(0),
    rawInfoProducer = cms.string('ecalTBunpack'),
    recInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    planeShift = cms.vdouble(0.0, 0.0, 0.525, 0.0217),
    zPosition = cms.vdouble(-3251.0, -2881.0, -751.0, -381.0),
    rawInfoCollection = cms.string('')
)


