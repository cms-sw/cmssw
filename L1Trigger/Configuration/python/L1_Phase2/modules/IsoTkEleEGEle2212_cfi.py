import FWCore.ParameterSet.Config as cms

IsoTkEleEGEle2212 = cms.EDFilter("L1GTDoubleObjectCond",
    collection1 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.479),
        regionsMaxRelIsolationPt = cms.vdouble(0.13, 0.28),
        regionsMinPt = cms.vdouble(17.9, 16.1),
        regionsQualityFlags = cms.vuint32(2, 2),
        tag = cms.InputTag("l1tGTProducer","CL2Electrons")
    ),
    collection2 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.479),
        regionsMinPt = cms.vdouble(6.2, 7.7),
        regionsQualityFlags = cms.vuint32(4, 4),
        tag = cms.InputTag("l1tGTProducer","CL2Photons")
    ),
    cos_phi_lut = cms.PSet(
        refToPSet_ = cms.string('l1tGTCosPhiLUT')
    ),
    cosh_eta_lut = cms.PSet(
        refToPSet_ = cms.string('l1tGTCoshEtaLUT')
    ),
    cosh_eta_lut2 = cms.PSet(
        refToPSet_ = cms.string('l1tGTCoshEtaLUT2')
    ),
    inv_mass_checks = cms.untracked.bool(False),
    minDR = cms.double(0.1),
    primVertTag = cms.InputTag("l1tGTProducer","GTTPrimaryVert"),
    sanity_checks = cms.untracked.bool(False),
    scales = cms.PSet(
        beta_lsb = cms.double(0.0625),
        eta_lsb = cms.double(0.0007669903939428206),
        isolationPT_lsb = cms.double(0.25),
        mass_lsb = cms.double(0.25),
        neg_chg = cms.int32(1),
        pT_lsb = cms.double(0.03125),
        phi_lsb = cms.double(0.0007669903939428206),
        pos_chg = cms.int32(0),
        scalarSumPT_lsb = cms.double(0.03125),
        seed_pT_lsb = cms.double(0.25),
        seed_z0_lsb = cms.double(0.05859375),
        sum_pT_pv_lsb = cms.double(0.25),
        z0_lsb = cms.double(0.000390625)
    )
)
