import FWCore.ParameterSet.Config as cms

#
from HLTrigger.special.PixelMBCommon_cff import *
import copy
from HLTrigger.special.hltPixlMBForAlignment_cfi import *
#   do the filter work 
hltPixelMBForAlignment = copy.deepcopy(hltPixlMBForAlignment)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#   HLT prescaler
preMBForAlignment = copy.deepcopy(hltPrescaler)
#   the full sequence
hltMBForAlignment = cms.Sequence(cms.SequencePlaceholder("hltBegin")+preMBForAlignment+l1seedMinBiasPixel+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("pixelTrackingForMinBias")+hltPixelCands*hltPixelMBForAlignment)
preMBForAlignment.prescaleFactor = 1

