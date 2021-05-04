import FWCore.ParameterSet.Config as cms

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices

offlinePrimaryVerticesRecovery = offlinePrimaryVertices.clone(

  isRecoveryIteration = True,
  recoveryVtxCollection = "offlinePrimaryVertices",

  TkFilterParameters = dict(
        maxNormalizedChi2 = 999.0,
        minPixelLayersWithHits= 0,
        minSiliconLayersWithHits = 0,
        maxD0Significance = 999.0, 
        minPt = 0.0,
        maxEta = 999.0,
        trackQuality = "any"
  ),

  TkClusParameters = dict(
        algorithm = "gap",
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(1.0)       
        )
  ),

  vertexCollections = [offlinePrimaryVertices.vertexCollections[0].clone()]
)    

from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi import offlineSlimmedPrimaryVertices
offlineSlimmedPrimaryVerticesRecovery = offlineSlimmedPrimaryVertices.clone(
    src = "offlinePrimaryVerticesRecovery",
    score = None
)
