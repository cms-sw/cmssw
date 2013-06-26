import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.candidateBenchmark_cfi import *
from DQMOffline.PFTau.pfCandidateBenchmark_cfi import *
from DQMOffline.PFTau.metBenchmark_cfi import *

DQMOfflineParticleFlowSequence = cms.Sequence (
    candidateBenchmark + 
    pfCandidateBenchmark + 
    metBenchmark +
    matchMetBenchmark
)
