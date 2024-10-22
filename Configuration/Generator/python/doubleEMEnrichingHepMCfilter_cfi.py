import FWCore.ParameterSet.Config as cms

doubleEMenrichingHepMCfilterParams = cms.PSet(
    # seed thresholds
    PtSeedThr         = cms.double(5.0),
    EtaSeedThr        = cms.double(2.8),

    # photon thresholds
    PtGammaThr        = cms.double(0.0),
    EtaGammaThr       = cms.double(2.8),

    # electron threshold
    PtElThr           = cms.double(2.0),
    EtaElThr          = cms.double(2.8),

    dRSeedMax         = cms.double(0.0),
    dPhiSeedMax       = cms.double(0.2),
    dEtaSeedMax       = cms.double(0.12),
    dRNarrowCone      = cms.double(0.02),
    
    PtTkThr           = cms.double(1.6),
    EtaTkThr          = cms.double(2.2),
    dRTkMax           = cms.double(0.2),

    PtMinCandidate1   = cms.double(15.0),
    PtMinCandidate2   = cms.double(15.0),
    EtaMaxCandidate   = cms.double(3.0),
    
    NTkConeMax        = cms.int32(2),
    NTkConeSum        = cms.int32(4),

    # mass 80..Inf GeV
    InvMassMin        = cms.double(80.0),
    InvMassMax        = cms.double(14000.0),

    EnergyCut         = cms.double(1.0),

    AcceptPrompts     = cms.bool(True),
    PromptPtThreshold = cms.double(15.0),
    )
