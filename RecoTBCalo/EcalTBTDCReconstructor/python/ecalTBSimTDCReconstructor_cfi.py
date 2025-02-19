import FWCore.ParameterSet.Config as cms

# Simulated TDC reconstruction
ecalTBSimTDCReconstructor = cms.EDProducer("EcalTBTDCRecInfoProducer",
    use2004OffsetConvention = cms.untracked.bool(False),
    eventHeaderProducer = cms.string('SimEcalEventHeader'),
    eventHeaderCollection = cms.string('EcalTBEventHeader'),
    rawInfoProducer = cms.string('simEcalUnsuppressedDigis'),
    recInfoCollection = cms.string('EcalTBTDCRecInfo'),
    tdcRanges = cms.VPSet(cms.PSet(
        endRun = cms.int32(999999),
        tdcMax = cms.vdouble(1008.0, 927.0, 927.0, 927.0, 927.0),
        startRun = cms.int32(-1),
        tdcMin = cms.vdouble(748.0, 400.0, 400.0, 400.0, 400.0)
    )),
    rawInfoCollection = cms.string('EcalTBTDCRawInfo')
)


