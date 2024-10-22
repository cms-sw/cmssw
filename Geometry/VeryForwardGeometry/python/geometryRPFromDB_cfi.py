import FWCore.ParameterSet.Config as cms

ctppsGeometryESModule = cms.ESProducer("CTPPSGeometryESModule",
    fromPreprocessedDB = cms.untracked.bool(True),
    verbosity = cms.untracked.uint32(1),
    isRun2 = cms.bool(False),
)

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018

(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(ctppsGeometryESModule, isRun2=True)
