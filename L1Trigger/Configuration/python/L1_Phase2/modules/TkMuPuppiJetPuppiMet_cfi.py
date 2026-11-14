import FWCore.ParameterSet.Config as cms

TkMuPuppiJetPuppiMet = cms.EDFilter("L1GTTripleObjectCond",
    collection1 = cms.PSet(
        maxEta = cms.double(2.1),
        maxPrimVertDz = cms.double(1),
        minEta = cms.double(-2.1),
        primVertex = cms.uint32(0),
        qualityFlags = cms.uint32(2),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 0.83, 1.24),
        regionsMinPt = cms.vdouble(3, 3, 3),
        tag = cms.InputTag("l1tGTProducer","GMTTkMuons")
    ),
    collection2 = cms.PSet(
        maxEta = cms.double(2.4),
        minEta = cms.double(-2.4),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 1.5, 2.4, 3.0),
        regionsMinPt = cms.vdouble(75.3, 62.7, 50.7, 86.8),
        tag = cms.InputTag("l1tGTProducer","CL2JetsSC4")
    ),
    collection3 = cms.PSet(
        minPt = cms.double(51.5),
        tag = cms.InputTag("l1tGTProducer","CL2EtSum")
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
