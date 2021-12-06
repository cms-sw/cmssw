import FWCore.ParameterSet.Config as cms
import RecoJets.JetProducers.BetaStarVarProducer_cfi as _mod

ak4BetaStar = _mod.BetaStarVarProducer.clone(
    srcJet = "slimmedJets",    
    srcPF  = "packedPFCandidates",
    maxDR  = 0.4
)
