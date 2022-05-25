import FWCore.ParameterSet.Config as cms

pfClustersFromL1EGClusters = cms.EDProducer("PFClusterProducerFromL1EGClusters",
    corrector = cms.string(''),
    etMin = cms.double(0.5),
    resol = cms.PSet(
        etaBins = cms.vdouble(0.7, 1.2, 1.6),
        kind = cms.string('calo'),
        offset = cms.vdouble(0.838, 0.924, 1.101),
        scale = cms.vdouble(0.012, 0.017, 0.018)
    ),
    src = cms.InputTag("L1EGammaClusterEmuProducer")
)
