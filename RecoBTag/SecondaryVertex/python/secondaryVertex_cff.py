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

# backwards compatibility

simpleSecondaryVertexBJetTags = simpleSecondaryVertexHighEffBJetTags.clone()
