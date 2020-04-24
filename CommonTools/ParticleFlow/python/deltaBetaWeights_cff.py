import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.deltaBetaWeights_cfi import *

pfDeltaBetaWeightingSequence = cms.Sequence(pfWeightedPhotons+pfWeightedNeutralHadrons)
