import FWCore.ParameterSet.Config as cms

from HLTrigger.special.PixelMBCommon_cff import *
import copy
from HLTrigger.special.hltPixlMBFilt_cfi import *
hltMinBiasPixelFilter = copy.deepcopy(hltPixlMBFilt)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
preMinBiasPixel = copy.deepcopy(hltPrescaler)
hltMinBiasPixel = cms.Sequence(cms.SequencePlaceholder("hltBegin")+preMinBiasPixel+l1seedMinBiasPixel+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("pixelTrackingForMinBias")+hltPixelCands*hltMinBiasPixelFilter)
preMinBiasPixel.prescaleFactor = 1

