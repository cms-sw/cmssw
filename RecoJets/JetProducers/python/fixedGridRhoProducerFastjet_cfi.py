import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.default_FixedGridRhoProducerFastjet_cfi import default_FixedGridRhoProducerFastjet
fixedGridRhoFastjetAll = default_FixedGridRhoProducerFastjet.clone(
    pfCandidatesTag = "particleFlow"
)

fixedGridRhoFastjetAllCalo =  default_FixedGridRhoProducerFastjet.clone(
    pfCandidatesTag = "towerMaker"
)
