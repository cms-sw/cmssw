import FWCore.ParameterSet.Config as cms

# HF RecoEcalCandidate Producer
hfRecoEcalCandidate = cms.EDProducer("HFRecoEcalCandidateProducer",
    e9e25Cut = cms.double(0.88),
    hfclusters = cms.untracked.InputTag("hfEMClusters"),
    intercept2DCut = cms.double(0.16),
    Correct = cms.bool(True)
)


