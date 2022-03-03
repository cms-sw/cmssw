import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018

from EventFilter.CTPPSRawToDigi.totemVFATRawToDigi_cfi import totemVFATRawToDigi

totemRPRawToDigi = totemVFATRawToDigi.clone(
    subSystem = cms.string('TrackingStrip'),
    fedIds = cms.vuint32(578, 580, 584, 585)
)

# for Run 2 backward compatibility
(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(totemRPRawToDigi,
fedIds = cms.vuint32())
