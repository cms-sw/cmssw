import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
#
# Simple Min. and Zero Bias Triggers
#
# Logic: 
# at HLT level: always Zero Bias - do not need Min Bias (100Hz/100kHz = 1 permille effect!)
# at L1T level: Zero Bias (to check L1) or Min Bias (to check HLT)
#
#
# Min. Bias
#
# HLT Level-1 Seed filter:
hltl1sMin = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#
# HLTPrescaler:
hltpreMin = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
#
# Zero Bias
#
# HLT Level-1 Seed filter:
hltl1sZero = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#
# HLTPrescaler:
hltpreZero = copy.deepcopy(hltPrescaler)
#
# Actual HLT path:
hltMinBias = cms.Sequence(cms.SequencePlaceholder("hltBegin")+hltl1sMin+hltpreMin)
#
# Actual HLT path:
hltZeroBias = cms.Sequence(cms.SequencePlaceholder("hltBegin")+hltl1sZero+hltpreZero)
hltl1sMin.L1SeedsLogicalExpression = 'L1_MinBias_HTT10'
hltpreMin.prescaleFactor = 1
hltl1sZero.L1SeedsLogicalExpression = 'L1_ZeroBias'
hltpreZero.prescaleFactor = 1

