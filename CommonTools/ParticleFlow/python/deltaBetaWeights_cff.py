import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.deltaBetaWeights_cfi import *

pfDeltaBetaWeightingTask = cms.Task(pfWeightedPhotons,pfWeightedNeutralHadrons)
pfDeltaBetaWeightingSequence = cms.Sequence(pfDeltaBetaWeightingTask)
