import FWCore.ParameterSet.Config as cms

from RecoBTag.Combined.combinedMVA_EventSetup_cff import *

# CombinedMVA V2
from RecoBTag.Combined.combinedMVAV2BJetTags_cfi import *
from RecoBTag.Combined.negativeCombinedMVAV2BJetTags_cfi import *
from RecoBTag.Combined.positiveCombinedMVAV2BJetTags_cfi import *
from RecoBTag.Combined.pfCombinedMVAV2BJetTags_cfi import *
from RecoBTag.Combined.pfNegativeCombinedMVAV2BJetTags_cfi import *
from RecoBTag.Combined.pfPositiveCombinedMVAV2BJetTags_cfi import *

# Charge tagger
from RecoBTag.Combined.pfChargeBJetTags_cfi import *

combinedMVATask = cms.Task(
    combinedMVAV2BJetTags,
    negativeCombinedMVAV2BJetTags,
    positiveCombinedMVAV2BJetTags,
    pfCombinedMVAV2BJetTags,
    pfNegativeCombinedMVAV2BJetTags,
    pfPositiveCombinedMVAV2BJetTags
)
