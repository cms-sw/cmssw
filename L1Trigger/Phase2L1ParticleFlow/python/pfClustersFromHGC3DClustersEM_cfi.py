import FWCore.ParameterSet.Config as cms

pfClustersFromHGC3DClustersEM = cms.EDProducer("PFClusterProducerFromHGC3DClusters",
    src = cms.InputTag("hgcalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering"),
    emOnly = cms.bool(True),
    etMin = cms.double(0.0), 
    corrector  = cms.string("L1Trigger/Phase2L1ParticleFlow/data/emcorr_hgc.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 1.900,  2.200,  2.500,  2.800,  2.950),
            offset  = cms.vdouble( 0.651,  0.608,  0.496,  0.532,  0.358),
            scale   = cms.vdouble( 0.030,  0.024,  0.024,  0.023,  0.041),
            kind    = cms.string('calo'),
    )
)
