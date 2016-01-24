import FWCore.ParameterSet.Config as cms

from RecoBTag.Combined.combinedMVA_EventSetup_cff import *

from RecoBTag.Combined.combinedMVABJetTags_cfi import *
from RecoBTag.Combined.negativeCombinedMVABJetTags_cfi import *
from RecoBTag.Combined.positiveCombinedMVABJetTags_cfi import *

# New candidate based fwk
from RecoBTag.Combined.pfCombinedMVABJetTags_cfi import *
from RecoBTag.Combined.pfNegativeCombinedMVABJetTags_cfi import *
from RecoBTag.Combined.pfPositiveCombinedMVABJetTags_cfi import *

# CombinedMVA V2
from RecoBTag.Combined.combinedMVAV2BJetTags_cfi import *
from RecoBTag.Combined.pfCombinedMVAV2BJetTags_cfi import *

# Charge tagger
from RecoBTag.Combined.pfChargeBJetTags_cfi import *
