import FWCore.ParameterSet.Config as cms

#TDC recInfo reconstruction
tdcReco = cms.EDProducer("EcalTBTDCRecInfoProducer",
    fitMethod = cms.int32(0),
    eventHeaderProducer = cms.string('source'),
    eventHeaderCollection = cms.string(''),
    rawInfoProducer = cms.string('source'),
    tdcMin = cms.vint32(430, 400, 400, 400, 400),
    tdcMax = cms.vint32(958, 927, 927, 927, 927),
    recInfoCollection = cms.string('EcalTBTDCRecInfo'),
    rawInfoCollection = cms.string('')
)


