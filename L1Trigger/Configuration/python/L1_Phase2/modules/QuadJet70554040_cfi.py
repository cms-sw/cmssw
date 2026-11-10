import FWCore.ParameterSet.Config as cms

QuadJet70554040 = cms.EDFilter("L1GTQuadObjectCond",
    collection1 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.5, 2.4, 3.0),
        regionsMinPt = cms.vdouble(41.4, 37.0, 25, 57.0),
        tag = cms.InputTag("l1tGTProducer","CL2JetsSC4")
    ),
    collection2 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.5, 2.4, 3.0),
        regionsMinPt = cms.vdouble(28.6, 27.4, 25, 45.8),
        tag = cms.InputTag("l1tGTProducer","CL2JetsSC4")
    ),
    collection3 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.5, 2.4, 3.0),
        regionsMinPt = cms.vdouble(25, 25, 25, 34.6),
        tag = cms.InputTag("l1tGTProducer","CL2JetsSC4")
    ),
    collection4 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.5, 2.4, 3.0),
        regionsMinPt = cms.vdouble(25, 25, 25, 34.6),
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
