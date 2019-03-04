import FWCore.ParameterSet.Config as cms

pfClustersFromL1EGClusters = cms.EDProducer("PFClusterProducerFromL1EGClusters",
    src = cms.InputTag("L1EGammaClusterEmuProducer","L1EGXtalClusterEmulator"),
    etMin = cms.double(0.5),
    corrector  = cms.string("L1Trigger/Phase2L1ParticleFlow/data/emcorr_barrel.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 0.873,  1.081,  1.563),
            scale   = cms.vdouble( 0.011,  0.015,  0.012),
            kind    = cms.string('calo'),
    )
)
