import FWCore.ParameterSet.Config as cms

twinMuxStage2Digis = cms.EDProducer("L1TTwinMuxRawToDigi",
    DTTM7_FED_Source = cms.InputTag("rawDataCollector"),
    amcsecmap = cms.untracked.vint64(20015998343868, 20015998343868, 20015998343868, 20015998343868, 20015998343868),
    debug = cms.untracked.bool(False),
    feds = cms.untracked.vint32(1395, 1391, 1390, 1393, 1394),
    wheels = cms.untracked.vint32(-2, -1, 0, 1, 2)
)
