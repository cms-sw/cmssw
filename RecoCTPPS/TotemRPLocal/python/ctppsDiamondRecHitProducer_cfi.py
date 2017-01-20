import FWCore.ParameterSet.Config as cms

ctppsDiamondRecHit = cms.EDProducer("CTPPSDiamondRecHitProducer",
    verbosity = cms.int32(0),
    digiTag = cms.InputTag("ctppsDiamondRawToDigi", 'TimingDiamond'),
    timeSliceNs = cms.double(25.0/1024.0),
    timeShift = cms.int32(0), #FIXME
)
