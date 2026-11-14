import FWCore.ParameterSet.Config as cms

TkElePuppiJetMinDR = cms.EDFilter("L1GTDoubleObjectCond",
    collection1 = cms.PSet(
        maxEta = cms.double(2.1),
        maxPrimVertDz = cms.double(1),
        minEta = cms.double(-2.1),
        primVertex = cms.uint32(0),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.479),
        regionsMinPt = cms.vdouble(23.0, 20.3),
        regionsQualityFlags = cms.vuint32(2, 2),
        tag = cms.InputTag("l1tGTProducer","CL2Electrons")
    ),
    collection2 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        minPt = cms.double(25),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.5, 2.4, 3.0),
        tag = cms.InputTag("l1tGTProducer","CL2JetsSC4")
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
    minDR = cms.double(0.3),
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
