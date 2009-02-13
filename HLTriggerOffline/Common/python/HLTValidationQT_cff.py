import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Common.HLTValidationQTExample_cfi import *

hltvalidationqt = cms.Sequence(
    hltQTExample
    )
