import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.jetBProbabilityComputer_cfi import *

# positiveOnlyJetBProbability btag computer
positiveOnlyJetBProbabilityComputer = jetBProbabilityComputer.clone(
    trackIpSign = cms.int32(1) ## 0 = use both, 1 = positive only, -1 = negative only
)
