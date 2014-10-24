import FWCore.ParameterSet.Config as cms

jetCoreClusterSplitter = cms.EDProducer("JetCoreClusterSplitter",
           pixelClusters         = cms.InputTag('siPixelCluster'),
           vertices              = cms.InputTag('offlinePrimaryVertices'),
           pixelCPE = cms.string( "PixelCPEGeneric" ),
           verbose     = cms.bool(False),
           ptMin = cms.double(200),
           cores = cms.InputTag("ak5CaloJets"),
           chargeFractionMin = cms.double(1.2),
           deltaRmax  = cms.double(0.05),
    )


