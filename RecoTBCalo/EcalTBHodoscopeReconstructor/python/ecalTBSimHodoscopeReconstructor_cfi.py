import FWCore.ParameterSet.Config as cms

# Simulated hodoscope reconstruction
ecalTBSimHodoscopeReconstructor = cms.EDProducer("EcalTBHodoscopeRecInfoProducer",
    fitMethod = cms.int32(0),
    rawInfoProducer = cms.string('SimEcalTBHodoscope'),
    recInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    # vdouble planeShift = { 0.0, 0.0, 0.4633, 0.1000 }
    planeShift = cms.vdouble(0.0, 0.0, 0.0, 0.0),
    zPosition = cms.vdouble(-3251.0, -2881.0, -751.0, -381.0),
    rawInfoCollection = cms.string('EcalTBHodoscopeRawInfo')
)


