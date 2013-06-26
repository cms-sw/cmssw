import FWCore.ParameterSet.Config as cms

#TDC recInfo reconstruction for H2 ECAL&HCAL TB
ecal2006TBH2TDCReconstructor = cms.EDProducer("EcalTBH2TDCRecInfoProducer",
    rawInfoProducer = cms.string('tbunpacker'),
    triggerDataProducer = cms.string('tbunpacker'),
    recInfoCollection = cms.string('EcalTBTDCRecInfo'),
    tdcZeros = cms.VPSet(cms.PSet(
        endRun = cms.int32(31031),
        tdcZero = cms.double(1050.5),
        startRun = cms.int32(27540)
    ), 
        cms.PSet(
            endRun = cms.int32(999999),
            tdcZero = cms.double(1058.5),
            startRun = cms.int32(31032)
        )),
    rawInfoCollection = cms.string(''),
    triggerDataCollection = cms.string('')
)


