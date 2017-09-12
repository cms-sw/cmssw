import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteDefaultProducer_cfi import ctppsLocalTrackLiteDefaultProducer

ctppsLocalTrackLiteProducer = ctppsLocalTrackLiteDefaultProducer.clone()

# enable the module for CTPPS era(s)
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toModify(
    ctppsLocalTrackLiteProducer,
    doNothing = cms.bool(False)
)
