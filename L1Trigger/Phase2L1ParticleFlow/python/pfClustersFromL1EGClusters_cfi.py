import FWCore.ParameterSet.Config as cms

pfClustersFromL1EGClusters = cms.EDProducer("PFClusterProducerFromL1EGClusters",
    src = cms.InputTag("l1EGammaCrystalsProducer","L1EGXtalClusterNoCuts"),
    etMin = cms.double(0.5),
    corrector  = cms.string("L1Trigger/Phase2L1ParticleFlow/data/ecorr.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 1.300,  1.700,  2.800,  3.200),
            offset  = cms.vdouble( 1.158,  1.545,  0.732,  0.551),
            scale   = cms.vdouble( 0.014,  0.024,  0.028,  0.031),
            kind    = cms.string('calo'),
    )
)
