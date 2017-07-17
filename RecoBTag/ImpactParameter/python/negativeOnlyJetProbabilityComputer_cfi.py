import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.jetProbabilityComputer_cfi import *

# negativeOnlyJetProbability btag computer
negativeOnlyJetProbabilityComputer = jetProbabilityComputer.clone(
    trackIpSign = cms.int32(-1) ## 0 = use both, 1 = positive only, -1 = negative only
)
