import FWCore.ParameterSet.Config as cms


from DQMOffline.PFTau.pfCandidateBenchmark_cfi import pfCandidateBenchmark
from DQMOffline.PFTau.candidateBenchmark_cfi import candidateBenchmark

pfCandidateBenchmarkSequence = cms.Sequence(
    pfCandidateBenchmark +
    candidateBenchmark 
    )
