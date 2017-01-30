import FWCore.ParameterSet.Config as cms

ctppsDiamondRecHit = cms.EDProducer('CTPPSDiamondRecHitProducer',
    digiTag = cms.InputTag('ctppsDiamondRawToDigi', 'TimingDiamond'),
    timeSliceNs = cms.double(25.0/1024.0),
    timeShift = cms.int32(0), # to be determined at calibration level, will be replaced by a map channel id -> time shift
)
