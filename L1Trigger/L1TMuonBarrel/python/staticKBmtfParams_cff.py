import FWCore.ParameterSet.Config as cms

kalmanParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonBarrelKalmanParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

staticKBmtfParams = cms.ESProducer(
    'L1TMuonBarrelKalmanParamsESProducer',
    fwVersion = cms.uint32(0x95030160),
    LUTsPath = cms.string("L1Trigger/L1TMuon/data/bmtf_luts/kalmanLUTs_v302.root")
)
