import FWCore.ParameterSet.Config as cms

tpScales = cms.PSet(
    HF=cms.PSet(
        NCTShift=cms.int32(1),
        RCTShift=cms.int32(3),
    ),
    HBHE=cms.PSet(
        LSBQIE8=cms.double(1/8.),
        LSBQIE11=cms.double(1/16.),
        LSBQIE11Overlap=cms.double(1/8.),
    )
)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toModify(tpScales.HF, NCTShift=2)

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(tpScales.HBHE, LSBQIE11Overlap=1/16.)
