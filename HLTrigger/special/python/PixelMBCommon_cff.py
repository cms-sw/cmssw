import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
#
#   include "HLTrigger/Configuration/data/common/HLTSetup.cff"
#    include "HLTrigger/special/data/HLTFullRecoForSpecial.cff"
l1seedMinBiasPixel = copy.deepcopy(hltLevel1GTSeed)
#   convert all pixel tracks to RecoChargedCandidates
hltPixelCands = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("pixelTracksForMinBias"),
    #      just pretend tracks are pi+, many will not really be...
    particleType = cms.string('pi+')
)

l1seedMinBiasPixel.L1SeedsLogicalExpression = 'L1_ZeroBias'

