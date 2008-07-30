import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.impactParameter_cff import *
import copy
from RecoBTag.ImpactParameter.impactParameter_cfi import *
hltBLifetimeL25TagInfos = copy.deepcopy(impactParameterTagInfos)
import copy
from RecoBTag.ImpactParameter.trackCountingHighEffBJetTags_cfi import *
hltBLifetimeL25BJetTags = copy.deepcopy(trackCountingHighEffBJetTags)
import copy
from RecoBTag.ImpactParameter.impactParameter_cfi import *
hltBLifetimeL3TagInfos = copy.deepcopy(impactParameterTagInfos)
import copy
from RecoBTag.ImpactParameter.trackCountingHighEffBJetTags_cfi import *
hltBLifetimeL3BJetTags = copy.deepcopy(trackCountingHighEffBJetTags)
hltBLifetimeL25TagInfos.jetTracks = 'hltBLifetimeL25Associator'
hltBLifetimeL25TagInfos.primaryVertex = 'pixelVertices'
hltBLifetimeL25TagInfos.computeProbabilities = False
hltBLifetimeL25TagInfos.minimumNumberOfHits = 3
hltBLifetimeL25BJetTags.tagInfo = 'hltBLifetimeL25TagInfos'
hltBLifetimeL3TagInfos.jetTracks = 'hltBLifetimeL3Associator'
hltBLifetimeL3TagInfos.primaryVertex = 'pixelVertices'
hltBLifetimeL3TagInfos.computeProbabilities = False
hltBLifetimeL3TagInfos.minimumNumberOfHits = 8
hltBLifetimeL3BJetTags.tagInfo = 'hltBLifetimeL3TagInfos'

