import FWCore.ParameterSet.Config as cms

from ..modules.ge0TriggerConvertedPseudoDigis_cfi import *
from ..modules.gemRecHits_cfi import *
from ..modules.gemSegments_cfi import *

ge0TriggerPseudoDigiTask = cms.Task(ge0TriggerConvertedPseudoDigis, gemRecHits, gemSegments)
