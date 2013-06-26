import FWCore.ParameterSet.Config as cms

#TDC recInfo reconstruction
ecal2007H4TBTDCReconstructor = cms.EDProducer("EcalTBTDCRecInfoProducer",
    use2004OffsetConvention = cms.untracked.bool(False),
    fitMethod = cms.int32(0),
    eventHeaderProducer = cms.string('ecalTBunpack'),
    eventHeaderCollection = cms.string(''),
    rawInfoProducer = cms.string('ecalTBunpack'),
    recInfoCollection = cms.string('EcalTBTDCRecInfo'),
    tdcRanges = cms.VPSet(cms.PSet(
        endRun = cms.int32(99999),
        tdcMax = cms.vdouble(1531.0, 927.0, 927.0, 927.0, 927.0),
        startRun = cms.int32(16585),
        tdcMin = cms.vdouble(1269.0, 400.0, 400.0, 400.0, 400.0)
    )),
    rawInfoCollection = cms.string('')
)


