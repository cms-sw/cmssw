import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.candidateJetBProbabilityComputer_cfi import *

# negativeOnlyJetBProbability btag computer
candidateNegativeOnlyJetBProbabilityComputer = candidateJetBProbabilityComputer.clone(
    trackIpSign = -1 ## 0 = use both, 1 = positive only, -1 = negative only
)
