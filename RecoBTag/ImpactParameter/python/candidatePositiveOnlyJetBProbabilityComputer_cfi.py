import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.candidateJetBProbabilityComputer_cfi import *

# positiveOnlyJetBProbability btag computer
candidatePositiveOnlyJetBProbabilityComputer = candidateJetBProbabilityComputer.clone(
    trackIpSign = 1 ## 0 = use both, 1 = positive only, -1 = negative only
)
# foo bar baz
# c02hrUsh4YKTG
# 0EH6Ntd8PbDoP
