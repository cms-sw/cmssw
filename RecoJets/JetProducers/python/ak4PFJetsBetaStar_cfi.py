import FWCore.ParameterSet.Config as cms


ak4BetaStar = cms.EDProducer("BetaStarPackedCandidateVarProducer",
    srcJet = cms.InputTag("slimmedJets"),    
    srcPF = cms.InputTag("packedPFCandidates"),
    maxDR = cms.double(0.4)
)
