import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.candidateJetProbabilityComputer_cfi import *

# positiveOnlyJetProbability btag computer
candidatePositiveOnlyJetProbabilityComputer = candidateJetProbabilityComputer.clone(
    trackIpSign = 1 ## 0 = use both, 1 = positive only, -1 = negative only
)
# foo bar baz
# 8crwWz5M52N4k
# UiNu3QJr2uVDC
