import FWCore.ParameterSet.Config as cms

# modify CTPPS 2016 raw-to-digi modules

from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import *

from EventFilter.CTPPSRawToDigi.ctppsPixelRawToDigi_cfi import ctppsPixelDigis
ctppsPixelDigis.inputLabel = cms.InputTag("ctppsPixelRawData")

ctppsRawToDigi = cms.Sequence(totemRPRawToDigi*totemTriggerRawToDigi*ctppsPixelDigis)
