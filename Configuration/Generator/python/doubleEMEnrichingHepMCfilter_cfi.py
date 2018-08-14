import FWCore.ParameterSet.Config as cms

doubleEMenrichingHepMCfilterParams = cms.PSet(
    # seed thresholds
    PtSeedThr         = cms.untracked.double(5.0),
    EtaSeedThr        = cms.untracked.double(2.8),

    # photon thresholds
    PtGammaThr        = cms.untracked.double(0.0),
    EtaGammaThr       = cms.untracked.double(2.8),

    # electron threshold
    PtElThr           = cms.untracked.double(2.0),
    EtaElThr          = cms.untracked.double(2.8),

    dRSeedMax         = cms.untracked.double(0.0),
    dPhiSeedMax       = cms.untracked.double(0.2),
    dEtaSeedMax       = cms.untracked.double(0.12),
    dRNarrowCone      = cms.untracked.double(0.02),
    
    PtTkThr           = cms.untracked.double(1.6),
    EtaTkThr          = cms.untracked.double(2.2),
    dRTkMax           = cms.untracked.double(0.2),

    PtMinCandidate1   = cms.untracked.double(15.0),
    PtMinCandidate2   = cms.untracked.double(15.0),
    EtaMaxCandidate   = cms.untracked.double(3.0),
    
    NTkConeMax        = cms.untracked.int32(2),
    NTkConeSum        = cms.untracked.int32(4),

    # mass 80..Inf GeV
    InvMassMin        = cms.untracked.double(80.0),
    InvMassMax        = cms.untracked.double(14000.0),

    EnergyCut         = cms.untracked.double(1.0),

    AcceptPrompts     = cms.untracked.bool(True),
    PromptPtThreshold = cms.untracked.double(15.0),
    )
