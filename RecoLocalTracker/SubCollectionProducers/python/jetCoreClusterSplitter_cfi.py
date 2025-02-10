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
           expSizeXAtLorentzAngleIncidence = cms.double(1.5),
           expSizeXDeltaPerTanAlpha = cms.double(0.0),
           expSizeYAtNormalIncidence = cms.double(1.3),
           tanLorentzAngle = cms.double(0.0), # doesn't really matter if expSizeXDeltaPerTanAlpha == 0
           tanLorentzAngleBarrelLayer1 = cms.double(0.0),
	

    )


