import FWCore.ParameterSet.Config as cms

from ..modules.ctppsDiamondRawToDigi_cfi import *
from ..modules.ctppsPixelDigis_cfi import *
from ..modules.totemRPRawToDigi_cfi import *
from ..modules.totemTimingRawToDigi_cfi import *
from ..modules.totemTriggerRawToDigi_cfi import *

ctppsRawToDigiTask = cms.Task(ctppsDiamondRawToDigi, ctppsPixelDigis, totemRPRawToDigi, totemTimingRawToDigi, totemTriggerRawToDigi)
