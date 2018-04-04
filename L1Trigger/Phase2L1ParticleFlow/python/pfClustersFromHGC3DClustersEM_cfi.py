import FWCore.ParameterSet.Config as cms

pfClustersFromHGC3DClustersEM = cms.EDProducer("PFClusterProducerFromHGC3DClusters",
    src = cms.InputTag("hgcalTriggerPrimitiveDigiProducer","cluster3D"),
    emOnly = cms.bool(True),
    etMin = cms.double(0.0), 
    corrector  = cms.string("L1Trigger/Phase2L1ParticleFlow/data/ecorr.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 1.300,  1.700,  2.800,  3.200),
            offset  = cms.vdouble( 1.158,  1.545,  0.732,  0.551),
            scale   = cms.vdouble( 0.014,  0.024,  0.028,  0.031),
            kind    = cms.string('calo'),
    )
)
