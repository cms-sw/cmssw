import FWCore.ParameterSet.Config as cms

# Configuration parameters for Mahi
mahiParameters = cms.PSet(

    calculateArrivalTime  = cms.bool(True),
    dynamicPed        = cms.bool(False),
    ts4Thresh         = cms.double(0.0),
    chiSqSwitch       = cms.double(15.0),
    activeBXs         = cms.vint32(-3, -2, -1, 0, 1, 2, 3, 4),
    nMaxItersMin      = cms.int32(500),
    nMaxItersNNLS     = cms.int32(500),
    deltaChiSqThresh  = cms.double(1e-3),
    nnlsThresh        = cms.double(1e-11)
)
