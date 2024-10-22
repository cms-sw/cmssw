import FWCore.ParameterSet.Config as cms
import RecoJets.JetProducers.FastjetJetProducer_cfi as _mod

hltak4CaloJets = _mod.FastjetJetProducer.clone(
    useDeterministicSeed = True,
    voronoiRfact     = 0.9,
    srcPVs           = "offlinePrimaryVertices",
    inputEtMin       = 0.3,
    jetPtMin         = 3.0,
    jetType          = 'CaloJet',
    radiusPU         = 0.4,
    doAreaDiskApprox = True,
    DzTrVtxMax       = 0.0,
    DxyTrVtxMax      = 0.0,
    src              = "hltTowerMakerForAll"
)
