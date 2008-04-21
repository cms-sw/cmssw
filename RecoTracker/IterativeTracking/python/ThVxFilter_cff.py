import FWCore.ParameterSet.Config as cms

import copy
from RecoParticleFlow.PFTracking.vertexFilter_cfi import *
thStep = copy.deepcopy(vertFilter)
thStep.recTracks = cms.InputTag("thWithMaterialTracks")
thStep.DistZFromVertex = 0.1
thStep.TrackAlgorithm = 'iter3'

