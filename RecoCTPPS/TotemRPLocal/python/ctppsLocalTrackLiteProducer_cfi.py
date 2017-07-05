import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteDefaultProducer_cfi import ctppsLocalTrackLiteDefaultProducer

ctppsLocalTrackLiteProducer = ctppsLocalTrackLiteDefaultProducer.clone()

# enable the module for CTPPS era(s)
from Configuration.StandardSequences.Eras import eras
eras.ctpps_2016.toModify(
    ctppsLocalTrackLiteProducer,
    doNothing = cms.bool(False)
)
