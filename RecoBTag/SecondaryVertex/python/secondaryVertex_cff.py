import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoBTau.JetTagComputer.jetTagRecord_cfi import *
from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertex2TrkES_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertex3TrkES_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexES_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexMVAES_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexMVABJetTags_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackES_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexSoftLeptonES_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexSoftLeptonBJetTags_cfi import *

# IVF
from RecoBTag.SecondaryVertex.inclusiveSecondaryVertexFinderTagInfos_cfi import *
from RecoBTag.SecondaryVertex.combinedInclusiveSecondaryVertexBJetTags_cfi import *
#from RecoBTag.SecondaryVertex.combinedIVFES_cfi import * #not yet using dedicated training, share CSV ones
from RecoBTag.SecondaryVertex.bVertexFilter_cfi import *
inclusiveSecondaryVerticesFiltered = bVertexFilter.clone()
inclusiveSecondaryVerticesFiltered.vertexFilter.multiplicityMin = 2
inclusiveSecondaryVerticesFiltered.secondaryVertices = cms.InputTag("inclusiveSecondaryVertices")

from RecoBTag.SecondaryVertex.bToCharmDecayVertexMerger_cfi import *
from RecoBTag.SecondaryVertex.simpleInclusiveSecondaryVertexBJetTags_cfi import *
from RecoBTag.SecondaryVertex.doubleVertex2TrkES_cfi import *
from RecoBTag.SecondaryVertex.doubleSecondaryVertexHighEffBJetTags_cfi import *

# Negative taggers
from RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexNegativeES_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexNegativeBJetTags_cfi import *

# Positive taggers
from RecoBTag.SecondaryVertex.combinedSecondaryVertexPositiveES_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexPositiveBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedInclusiveSecondaryVertexPositiveBJetTags_cfi import *

# Backwards compatibility

simpleSecondaryVertexBJetTags = simpleSecondaryVertexHighEffBJetTags.clone()
