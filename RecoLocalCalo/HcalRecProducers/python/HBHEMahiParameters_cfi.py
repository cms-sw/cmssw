import FWCore.ParameterSet.Config as cms

# Configuration parameters for Mahi
mahiParameters = cms.PSet(

    calculateArrivalTime  = cms.bool(True),
    timeAlgo          = cms.int32(2), # 1=MahiTime, 2=ccTime
    thEnergeticPulses = cms.double(5.),
    dynamicPed        = cms.bool(False),
    ts4Thresh         = cms.double(0.0),
    chiSqSwitch       = cms.double(15.0),
    activeBXs         = cms.vint32(-3, -2, -1, 0, 1, 2, 3, 4),
    nMaxItersMin      = cms.int32(500),
    nMaxItersNNLS     = cms.int32(500),
    deltaChiSqThresh  = cms.double(1e-3),
    nnlsThresh        = cms.double(1e-11)
)

from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
#--- >= Run3 modification:
(run3_HB & run2_HE_2017).toModify(mahiParameters, chiSqSwitch = -1)
