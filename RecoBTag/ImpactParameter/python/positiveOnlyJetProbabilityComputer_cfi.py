import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.jetProbabilityComputer_cfi import *

# positiveOnlyJetProbability btag computer
positiveOnlyJetProbabilityComputer = jetProbabilityComputer.clone(
    trackIpSign = 1 ## 0 = use both, 1 = positive only, -1 = negative only
)
