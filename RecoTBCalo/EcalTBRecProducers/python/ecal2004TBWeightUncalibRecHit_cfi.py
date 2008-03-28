import FWCore.ParameterSet.Config as cms

ecal2004TBWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    use2004OffsetConvention = cms.untracked.bool(True),
    EBdigiCollection = cms.string(''),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    digiProducer = cms.string('source'),
    nbTimeBin = cms.int32(25),
    tdcRecInfoProducer = cms.string('ecal2004TBTDCReconstructor')
)


