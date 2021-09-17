import FWCore.ParameterSet.Config as cms

pfClustersFromHGC3DClusters = cms.EDProducer("PFClusterProducerFromHGC3DClusters",
    src = cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    corrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hadcorr_HGCal3D_TC.root"),
    correctorEmfMax = cms.double(1.125),
    preEmId  = cms.string("hOverE < 0.3 && hOverE >= 0"),
    emVsPionID = cms.PSet(
        isPUFilter = cms.bool(False),
        preselection = cms.string(""),
        method = cms.string("BDT"), # "" to be disabled, "BDT" to be enabled
        variables = cms.VPSet(
            cms.PSet(name = cms.string("fabs(eta)"), value = cms.string("abs(eta())")),
            cms.PSet(name = cms.string("coreShowerLength"), value = cms.string("coreShowerLength()")),
            cms.PSet(name = cms.string("maxLayer"), value = cms.string("maxLayer()")),
            cms.PSet(name = cms.string("hOverE"), value = cms.string("hOverE()")),
            cms.PSet(name = cms.string("sigmaZZ"), value = cms.string("sigmaZZ()")),
        ),
        weightsFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_vs_Pion_BDTweights.xml.gz"),
        wp = cms.string("0.01")
    ),
    emVsPUID = cms.PSet(
        isPUFilter = cms.bool(True),
        preselection = cms.string(""),
        method = cms.string("BDT"), # "" to be disabled, "BDT" to be enabled
        variables = cms.VPSet(
            cms.PSet(name = cms.string("fabs(eta)"), value = cms.string("abs(eta())")),
            cms.PSet(name = cms.string("coreShowerLength"), value = cms.string("coreShowerLength()")),
            cms.PSet(name = cms.string("maxLayer"), value = cms.string("maxLayer()")),
            cms.PSet(name = cms.string("sigmaPhiPhiTot"), value = cms.string("sigmaPhiPhiTot()")),
        ),
        weightsFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_Pion_vs_Neutrino_BDTweights.xml.gz"),
        wp = cms.string("-0.02")
    ),
    emOnly = cms.bool(False),
    etMin = cms.double(1.0),
    resol = cms.PSet(
        etaBins = cms.vdouble( 1.900,  2.200,  2.500,  2.800,  2.950),
        offset  = cms.vdouble( 2.593,  3.089,  2.879,  2.664,  2.947),
        scale   = cms.vdouble( 0.120,  0.098,  0.099,  0.098,  0.124),
        kind    = cms.string('calo')
    ),
)


from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11
phase2_hgcalV10.toModify(pfClustersFromHGC3DClusters,
    corrector = "L1Trigger/Phase2L1ParticleFlow/data/hadcorr_HGCal3D_TC_106X.root",
    resol = cms.PSet(
        etaBins = cms.vdouble( 1.700,  1.900,  2.200,  2.500,  2.800,  2.900),
        offset  = cms.vdouble(-0.819,  0.900,  2.032,  2.841,  2.865,  1.237),
        scale   = cms.vdouble( 0.320,  0.225,  0.156,  0.108,  0.119,  0.338),
        kind    = cms.string('calo')
    ),
) 
phase2_hgcalV11.toModify(pfClustersFromHGC3DClusters,
    corrector = "L1Trigger/Phase2L1ParticleFlow/data/hadcorr_HGCal3D_TC_110X.root",
    resol = cms.PSet(
        etaBins = cms.vdouble( 1.700,  1.900,  2.200,  2.500,  2.800,  2.900),
        offset  = cms.vdouble( 1.793,  1.827,  2.363,  2.538,  2.812,  2.642),
        scale   = cms.vdouble( 0.138,  0.137,  0.124,  0.115,  0.106,  0.121),
        kind    = cms.string('calo'),
    ),
) 
