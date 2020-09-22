import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.candidateJetProbabilityComputer_cfi import *

# negativeOnlyJetProbability btag computer
candidateNegativeOnlyJetProbabilityComputer = candidateJetProbabilityComputer.clone(
    trackIpSign = -1 ## 0 = use both, 1 = positive only, -1 = negative only
)


