import FWCore.ParameterSet.Config as cms

hltEgammaHGCALIDVarsUnseeded = cms.EDProducer("EgammaHLTHGCalIDVarProducer",
    hgcalRecHits = cms.InputTag("particleFlowRecHitHGC"),
    layerClusters = cms.InputTag("hgcalMergeLayerClusters"),
    rCylinder = cms.double(2.8),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded")
)
