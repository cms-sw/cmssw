import FWCore.ParameterSet.Config as cms

from RecoBTag.Combined.combinedMVAComputer_cfi import *
from RecoBTag.Combined.negativeCombinedMVAComputer_cfi import *
from RecoBTag.Combined.positiveCombinedMVAComputer_cfi import *

# New candidate based fwk
from RecoBTag.Combined.candidateCombinedMVAComputer_cfi import *
from RecoBTag.Combined.candidateNegativeCombinedMVAComputer_cfi import *
from RecoBTag.Combined.candidatePositiveCombinedMVAComputer_cfi import *

# CombinedMVA V2
from RecoBTag.Combined.combinedMVAV2Computer_cfi import *
from RecoBTag.Combined.candidateCombinedMVAV2Computer_cfi import *

# Charge tagger
from RecoBTag.Combined.candidateChargeBTagComputer_cfi import *
