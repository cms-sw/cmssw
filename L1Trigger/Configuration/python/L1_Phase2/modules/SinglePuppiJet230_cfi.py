import FWCore.ParameterSet.Config as cms

SinglePuppiJet230 = cms.EDFilter("L1GTSingleObjectCond",
    maxEta = cms.double(2.4),
    minEta = cms.double(-2.4),
    primVertTag = cms.InputTag("l1tGTProducer","GTTPrimaryVert"),
    regionsAbsEtaLowerBounds = cms.vdouble(0, 1.5, 2.4, 3.0),
    regionsMinPt = cms.vdouble(176.9, 139.6, 149.1, 176.4),
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
    ),
    tag = cms.InputTag("l1tGTProducer","CL2JetsSC4")
)
