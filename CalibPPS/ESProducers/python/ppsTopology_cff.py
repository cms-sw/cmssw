import FWCore.ParameterSet.Config as cms

from CalibPPS.ESProducers.ppsPixelTopologyESSource_cfi import ppsPixelTopologyESSource

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018

(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(ppsPixelTopologyESSource, RunType = cms.string('Run2'), simYWidth = cms.double(24.4) )
