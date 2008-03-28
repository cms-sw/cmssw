import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#
# Prescaler
#
prescaleTriggerType = copy.deepcopy(hltPrescaler)
#
# Filter (moved to HLTFullRecoForSpecial.cff)
#
# module  filterTriggerType = triggerTypeFilter from "HLTrigger/special/data/TriggerTypeFilter.cfi"
# replace filterTriggerType.InputLabel = rawDataCollector
#
# resulting Sequence without L1 seed filter
#
sequenceTriggerType = cms.Sequence(cms.SequencePlaceholder("hltBegin")+prescaleTriggerType+cms.SequencePlaceholder("filterTriggerType"))
prescaleTriggerType.prescaleFactor = 1

