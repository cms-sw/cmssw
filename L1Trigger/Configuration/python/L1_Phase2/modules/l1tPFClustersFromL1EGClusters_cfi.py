import FWCore.ParameterSet.Config as cms

l1tPFClustersFromL1EGClusters = cms.EDProducer("PFClusterProducerFromL1EGClusters",
    corrector = cms.string(''),
    digi_src = cms.InputTag("l1tPhase2L1CaloEGammaEmulator","GCTDigitizedClusterToCorrelator"),
    etMin = cms.double(0.5),
    etaBounds = cms.vdouble(-1.5, 0.0, 1.5),
    maxClustersEtaPhi = cms.vuint32(
        8, 8, 8, 8, 8,
        8, 8, 8, 8, 8,
        8, 8, 8, 8, 8,
        8, 8, 8, 8, 8,
        8, 8, 8, 8, 8,
        8, 8, 8, 8, 8,
        8, 8, 8, 8, 8,
        8
    ),
    phiBounds = cms.vdouble(
        -3.14159265359, -2.792526803191111, -2.4434609527922224, -2.0943951023933334, -1.7453292519944446,
        -1.3962634015955555, -1.0471975511966667, -0.6981317007977778, -0.3490658503988889, 0.0,
        0.3490658503988889, 0.6981317007977778, 1.0471975511966667, 1.3962634015955555, 1.7453292519944446,
        2.0943951023933334, 2.4434609527922224, 2.792526803191111, 3.14159265359
    ),
    resol = cms.PSet(
        etaBins = cms.vdouble(0.7, 1.2, 1.6),
        kind = cms.string('calo'),
        offset = cms.vdouble(0.838, 0.924, 1.101),
        scale = cms.vdouble(0.012, 0.017, 0.018)
    ),
    src = cms.InputTag("l1tPhase2L1CaloEGammaEmulator","GCTClusters")
)
