import FWCore.ParameterSet.Config as cms

import L1Trigger.Phase2L1ParticleFlow.pfClustersFromHGC3DClusters_cfi

pfClustersFromHGC3DClustersEM = L1Trigger.Phase2L1ParticleFlow.pfClustersFromHGC3DClusters_cfi.pfClustersFromHGC3DClusters.clone(
    emOnly = cms.bool(True),
    useEMInterpretation = cms.string("emOnly"), # use EM intepretation to redefine the energy
    etMin = cms.double(0.0), 
    corrector  = cms.string("L1Trigger/Phase2L1ParticleFlow/data/emcorr_hgc.root"),
    preEmId  = cms.string(""),
    resol = cms.PSet(
            etaBins = cms.vdouble( 1.900,  2.200,  2.500,  2.800,  2.950),
            offset  = cms.vdouble( 0.566,  0.557,  0.456,  0.470,  0.324),
            scale   = cms.vdouble( 0.030,  0.024,  0.024,  0.023,  0.042),
            kind    = cms.string('calo'),
    )
)


from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11
phase2_hgcalV10.toModify(pfClustersFromHGC3DClustersEM,
    corrector = "L1Trigger/Phase2L1ParticleFlow/data/emcorr_hgc_106X.root",
    resol = cms.PSet(
        etaBins = cms.vdouble( 1.700,  1.900,  2.200,  2.500,  2.800,  2.900),
        offset  = cms.vdouble( 2.579,  2.176,  1.678,  0.911,  0.672, -2.292),
        scale   = cms.vdouble( 0.048,  0.026,  0.012,  0.016,  0.022,  0.538),
        kind    = cms.string('calo')
    ),
) 
phase2_hgcalV11.toModify(pfClustersFromHGC3DClustersEM,
    corrector = "L1Trigger/Phase2L1ParticleFlow/data/emcorr_hgc_110X.root",
    resol = cms.PSet(
        etaBins = cms.vdouble( 1.700,  1.900,  2.200,  2.500,  2.800,  2.900),
        offset  = cms.vdouble( 2.581,  2.289,  1.674,  0.927,  0.604, -2.377),
        scale   = cms.vdouble( 0.046,  0.025,  0.016,  0.017,  0.023,  0.500),
        kind    = cms.string('calo')
    ),
)
