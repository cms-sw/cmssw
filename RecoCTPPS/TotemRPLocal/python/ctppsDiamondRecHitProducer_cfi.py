import FWCore.ParameterSet.Config as cms

ctppsDiamondRecHit = cms.EDProducer("CTPPSDiamondRecHitProducer",
    verbosity = cms.int32(0),
    digiTag = cms.InputTag("ctppsDiamondRawToDigi", 'TimingDiamond'),
    timeSliceTons = cms.double(25.0/1024.0*1000.0),
    timeShift = cms.int32(0), #FIXME
)
