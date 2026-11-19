import FWCore.ParameterSet.Config as cms

hltEgammaHGCALIDVarsUnseeded = cms.EDProducer("EgammaHLTHGCalIDVarProducer",
    ticlGeom = cms.ESInputTag('', 'hlt'),
    hgcalRecHits = cms.InputTag("hltParticleFlowRecHitHGC"),
    layerClusters = cms.InputTag("hltMergeLayerClusters"),
    rCylinder = cms.double(2.8),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded")
)
