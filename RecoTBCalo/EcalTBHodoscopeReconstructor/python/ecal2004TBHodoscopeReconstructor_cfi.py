import FWCore.ParameterSet.Config as cms

#Hodoscope reconstruction
ecal2004TBHodoscopeReconstructor = cms.EDProducer("EcalTBHodoscopeRecInfoProducer",
    fitMethod = cms.int32(0),
    rawInfoProducer = cms.string('source'),
    recInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    planeShift = cms.vdouble(0.0, 0.0, 0.4633, 0.1),
    zPosition = cms.vdouble(-3251.0, -2881.0, -751.0, -381.0),
    rawInfoCollection = cms.string('')
)


