import FWCore.ParameterSet.Config as cms

from EventFilter.RawDataCollector.default_rawDataCollectorByLabel_cfi import default_rawDataCollectorByLabel as _rawDataCollectorByLabel
rawDataCollector = _rawDataCollectorByLabel.clone()
#
# Make changes if using the Stage 1 trigger
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("l1tDigiToRaw")) )
