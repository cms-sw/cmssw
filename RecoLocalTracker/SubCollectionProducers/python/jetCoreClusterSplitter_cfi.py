import FWCore.ParameterSet.Config as cms

jetCoreClusterSplitter = cms.EDProducer("JetCoreClusterSplitter",
           pixelClusters         = cms.InputTag('siPixelCluster'),
           vertices              = cms.InputTag('offlinePrimaryVertices'),
           pixelCPE = cms.string( "PixelCPEGeneric" ),
           verbose     = cms.bool(False),
           ptMin = cms.double(200),
           cores = cms.InputTag("ak5CaloJets"),
           chargeFractionMin = cms.double(2.0),
           deltaRmax  = cms.double(0.05),
           forceXError  = cms.double(100), #negative means do not force
           forceYError  = cms.double(150),
           fractionalWidth  = cms.double(0.4),
           chargePerUnit  = cms.double(2000),
           centralMIPCharge  = cms.double(26000),
	

    )


