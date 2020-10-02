import FWCore.ParameterSet.Config as cms

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices

offlinePrimaryVerticesRecovery = offlinePrimaryVertices.clone(

  isRecoveryIteration = cms.bool(True),
  recoveryVtxCollection = cms.InputTag("offlinePrimaryVertices"),

  TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(999.0),
        minPixelLayersWithHits=cms.int32(0),
        minSiliconLayersWithHits = cms.int32(0),
        maxD0Significance = cms.double(999.0), 
        minPt = cms.double(0.0),
        maxEta = cms.double(999.0),
        trackQuality = cms.string("any")
  ),

  TkClusParameters = cms.PSet(
        algorithm = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(1.0)        
        )
  ),

  vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               chi2cutoff = cms.double(2.5),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               ),
      ]
    )

)    

from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi import offlineSlimmedPrimaryVertices
offlineSlimmedPrimaryVerticesRecovery = cms.EDProducer("PATVertexSlimmer",
  src = cms.InputTag("offlinePrimaryVerticesRecovery"),
)

#offlineSlimmedPrimaryVerticesRecovery = offlineSlimmedPrimaryVertices.clone(
#    src = cms.InputTag("offlinePrimaryVerticesRecovery"),
#    score = cms.InputTag(""),
#)                            
