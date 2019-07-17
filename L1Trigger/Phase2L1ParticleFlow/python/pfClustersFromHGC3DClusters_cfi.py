import FWCore.ParameterSet.Config as cms

pfClustersFromHGC3DClusters = cms.EDProducer("PFClusterProducerFromHGC3DClusters",
    src = cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    corrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hadcorr_HGCal3D_TC.root"),
    correctorEmfMax = cms.double(1.125),
    emId  = cms.string("hOverE < 0.3 && hOverE >= 0"),
    emOnly = cms.bool(False),
    etMin = cms.double(1.0), 
    resol = cms.PSet(
        etaBins = cms.vdouble( 1.900,  2.200,  2.500,  2.800,  2.950),
        offset  = cms.vdouble( 2.889,  3.215,  3.238,  2.979,  3.333),
        scale   = cms.vdouble( 0.128,  0.111,  0.108,  0.110,  0.123),
        kind    = cms.string('calo')
    ),
)
