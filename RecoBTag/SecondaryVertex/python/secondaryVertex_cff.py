import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertex_EventSetup_cff import *

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackBJetTags_cfi import *

# IVF
from RecoBTag.SecondaryVertex.inclusiveSecondaryVertexFinderTagInfos_cfi import *
from RecoBTag.SecondaryVertex.combinedInclusiveSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.inclusiveSecondaryVerticesFiltered_cfi import *
from RecoBTag.SecondaryVertex.bToCharmDecayVertexMerger_cfi import *
from RecoBTag.SecondaryVertex.inclusiveSecondaryVertexFinderFilteredTagInfos_cfi import *
from RecoBTag.SecondaryVertex.simpleInclusiveSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.simpleInclusiveSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.doubleSecondaryVertexHighEffBJetTags_cfi import *

# Negative taggers
from RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.inclusiveSecondaryVertexFinderNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.inclusiveSecondaryVertexFinderFilteredNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleInclusiveSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleInclusiveSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeCombinedInclusiveSecondaryVertexV2BJetTags_cfi import *

# Positive taggers
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.positiveCombinedInclusiveSecondaryVertexV2BJetTags_cfi import *

# New candidate based fwk
from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfSimpleSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfSimpleSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfSimpleInclusiveSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfSimpleInclusiveSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfCombinedInclusiveSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfGhostTrackVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfGhostTrackBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderAK8TagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfBoostedDoubleSVAK8TagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfBoostedDoubleSecondaryVertexAK8BJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderCA15TagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfBoostedDoubleSVCA15TagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfBoostedDoubleSecondaryVertexCA15BJetTags_cfi import *

# Negative taggers
from RecoBTag.SecondaryVertex.pfSecondaryVertexNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.pfNegativeSimpleSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfNegativeSimpleSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfNegativeCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfNegativeSimpleInclusiveSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfNegativeSimpleInclusiveSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags_cfi import *

# Positive taggers
from RecoBTag.SecondaryVertex.pfPositiveCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.pfPositiveCombinedInclusiveSecondaryVertexV2BJetTags_cfi import *

secondaryVertexTask = cms.Task(
    secondaryVertexTagInfos,
    simpleSecondaryVertexHighEffBJetTags,
    simpleSecondaryVertexHighPurBJetTags,
    combinedSecondaryVertexV2BJetTags,
    ghostTrackVertexTagInfos,
    ghostTrackBJetTags,

    inclusiveSecondaryVertexFinderTagInfos,
    combinedInclusiveSecondaryVertexV2BJetTags,
    inclusiveSecondaryVerticesFiltered,
    bToCharmDecayVertexMerged,
    inclusiveSecondaryVertexFinderFilteredTagInfos,
    simpleInclusiveSecondaryVertexHighEffBJetTags,
    simpleInclusiveSecondaryVertexHighPurBJetTags,
    doubleSecondaryVertexHighEffBJetTags,

    secondaryVertexNegativeTagInfos,
    inclusiveSecondaryVertexFinderNegativeTagInfos,
    inclusiveSecondaryVertexFinderFilteredNegativeTagInfos,
    negativeSimpleSecondaryVertexHighEffBJetTags,
    negativeSimpleSecondaryVertexHighPurBJetTags,
    negativeSimpleInclusiveSecondaryVertexHighEffBJetTags,
    negativeSimpleInclusiveSecondaryVertexHighPurBJetTags,
    negativeCombinedSecondaryVertexV2BJetTags,
    negativeCombinedInclusiveSecondaryVertexV2BJetTags,

    positiveCombinedSecondaryVertexV2BJetTags,
    positiveCombinedInclusiveSecondaryVertexV2BJetTags,

    pfSecondaryVertexTagInfos,
    pfSimpleSecondaryVertexHighEffBJetTags,
    pfSimpleSecondaryVertexHighPurBJetTags,
    pfCombinedSecondaryVertexV2BJetTags,
    pfInclusiveSecondaryVertexFinderTagInfos,
    pfSimpleInclusiveSecondaryVertexHighEffBJetTags,
    pfSimpleInclusiveSecondaryVertexHighPurBJetTags,
    pfCombinedInclusiveSecondaryVertexV2BJetTags,
    pfGhostTrackVertexTagInfos,
    pfGhostTrackBJetTags,
    pfInclusiveSecondaryVertexFinderAK8TagInfos,
    pfBoostedDoubleSVAK8TagInfos,
    pfBoostedDoubleSecondaryVertexAK8BJetTags,
    pfInclusiveSecondaryVertexFinderCA15TagInfos,
    pfBoostedDoubleSVCA15TagInfos,
    pfBoostedDoubleSecondaryVertexCA15BJetTags,

    pfSecondaryVertexNegativeTagInfos,
    pfInclusiveSecondaryVertexFinderNegativeTagInfos,
    pfNegativeSimpleSecondaryVertexHighEffBJetTags,
    pfNegativeSimpleSecondaryVertexHighPurBJetTags,
    pfNegativeCombinedSecondaryVertexV2BJetTags,
    pfNegativeSimpleInclusiveSecondaryVertexHighEffBJetTags,
    pfNegativeSimpleInclusiveSecondaryVertexHighPurBJetTags,
    pfNegativeCombinedInclusiveSecondaryVertexV2BJetTags,

    pfPositiveCombinedSecondaryVertexV2BJetTags,
    pfPositiveCombinedInclusiveSecondaryVertexV2BJetTags
)
