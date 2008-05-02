import FWCore.ParameterSet.Config as cms

import copy
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
hltBLifetimeL25Associator = copy.deepcopy(ic5JetTracksAssociatorAtVertex)
import copy
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
hltBLifetimeL3Associator = copy.deepcopy(ic5JetTracksAssociatorAtVertex)
hltBLifetimeL25Associator.tracks = 'pixelTracks'
hltBLifetimeL25Associator.jets = 'hltBLifetimeL25Jets'
hltBLifetimeL3Associator.tracks = 'hltBLifetimeRegionalCtfWithMaterialTracks'
hltBLifetimeL3Associator.jets = 'hltBLifetimeL3Jets'

