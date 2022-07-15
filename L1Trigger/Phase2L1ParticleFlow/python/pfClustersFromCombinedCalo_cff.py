import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.pfClustersFromCombinedCalo_cfi import pfClustersFromCombinedCalo

# Using phase2_hgcalV10 to customize the config for all 106X samples, since there's no other modifier for it
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11

# Calorimeter part: ecal + hcal + hf only
pfClustersFromCombinedCaloHCal = pfClustersFromCombinedCalo.clone(
    hcalHGCTowers = [], hcalDigis = [],
    hcalDigisBarrel = True, hcalDigisHF = False,
    hadCorrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 2.582,  2.191, -0.077),
            scale   = cms.vdouble( 0.122,  0.143,  0.465),
            kind    = cms.string('calo'),
    ))
phase2_hgcalV10.toModify(pfClustersFromCombinedCaloHCal,
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel_106X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 3.084,  2.715,  0.107),
            scale   = cms.vdouble( 0.118,  0.130,  0.442),
            kind    = cms.string('calo'),
    )
)
phase2_hgcalV11.toModify(pfClustersFromCombinedCaloHCal,
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel_110X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 2.909,  2.864,  0.294),
            scale   = cms.vdouble( 0.119,  0.127,  0.442),
            kind    = cms.string('calo'),
    )
)

pfClustersFromCombinedCaloHF = pfClustersFromCombinedCalo.clone(
    ecalCandidates = [], hcalHGCTowers = [],
    phase2barrelCaloTowers = [],
    hadCorrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hfcorr.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 3.500,  4.000,  4.500,  5.000),
            offset  = cms.vdouble( 1.099,  0.930,  1.009,  1.369),
            scale   = cms.vdouble( 0.152,  0.151,  0.144,  0.179),
            kind    = cms.string('calo'),
    ))
phase2_hgcalV10.toModify(pfClustersFromCombinedCaloHF,
    hcalCandidates = cms.VInputTag(cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")),
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hfcorr_106X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 3.500,  4.000,  4.500,  5.000),
            offset  = cms.vdouble(-0.846,  0.696,  1.313,  1.044),
            scale   = cms.vdouble( 0.815,  0.164,  0.146,  0.192),
            kind    = cms.string('calo'),
    )
)
phase2_hgcalV11.toModify(pfClustersFromCombinedCaloHF,
    hcalCandidates = cms.VInputTag(cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")),
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hfcorr_110X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 3.500,  4.000,  4.500,  5.000),
            offset  = cms.vdouble(-1.125,  1.220,  1.514,  1.414),
            scale   = cms.vdouble( 0.868,  0.159,  0.148,  0.194),
            kind    = cms.string('calo'),
    )
)
