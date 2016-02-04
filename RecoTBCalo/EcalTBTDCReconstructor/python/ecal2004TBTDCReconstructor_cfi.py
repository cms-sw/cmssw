import FWCore.ParameterSet.Config as cms

#TDC recInfo reconstruction
ecal2004TBTDCReconstructor = cms.EDProducer("EcalTBTDCRecInfoProducer",
    use2004OffsetConvention = cms.untracked.bool(True),
    fitMethod = cms.int32(0),
    eventHeaderProducer = cms.string('source'),
    eventHeaderCollection = cms.string(''),
    rawInfoProducer = cms.string('source'),
    recInfoCollection = cms.string('EcalTBTDCRecInfo'),
    tdcRanges = cms.VPSet(cms.PSet(
        endRun = cms.int32(999999),
        tdcMax = cms.vdouble(958.0, 927.0, 927.0, 927.0, 927.0),
        startRun = cms.int32(-1),
        tdcMin = cms.vdouble(430.0, 400.0, 400.0, 400.0, 400.0)
    )),
    rawInfoCollection = cms.string('')
)


