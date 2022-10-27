import FWCore.ParameterSet.Config as cms

HTDisplacedJetsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_HT425_v", # Claimed path for Run3
        "HLT_HT430_DisplacedDijet40_DisplacedTrack_v", # Claimed path for Run3
        "HLT_HT650_DisplacedDijet60_Inclusive_v", # Claimed path for Run3
        "HLT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_v", # New path for Run 3 (introduced in HLT V1.1)
        "HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v", # New path for Run 3 (introduced in HLT V1.1)
        "HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v", # New path for Run 3 (introduced in HLT V1.1)
        "HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v", # New path for Run 3 (introduced in HLT V1.1)
        "HLT_HT170_L1SingleLLPJet_DisplacedDijet40_DisplacedTrack_v", # New path for Run 3 (introduced in HLT V1.3)
        "HLT_HT320_L1SingleLLPJet_DisplacedDijet60_Inclusive_v", # New path for Run 3 (introduced in HLT V1.3)
        "HLT_HT200_L1SingleLLPJet_DelayedJet40_SingleDelay1nsTrackless_v", # New path for Run 3 (introduced in HLT V1.3)
        "HLT_HT200_L1SingleLLPJet_DelayedJet40_DoubleDelay0p5nsTrackless_v", # New path for Run 3 (introduced in HLT V1.3)
        "HLT_HT200_L1SingleLLPJet_DisplacedDijet30_Inclusive1PtrkShortSig5_v", # New path for Run 3 (introduced in HLT V1.3)
        ),
    recPFMHTLabel  = cms.InputTag("recoExoticaValidationHT"),
    recPFJetLabel  = cms.InputTag("ak4PFJets"),
    # -- Analysis specific cuts
    MET_genCut      = cms.string("sumEt > 75"),
    MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150,
                                   200, 220, 240, 260, 280, 300,
                                   320, 340, 360, 380, 400, 420,
                                   440, 460, 480, 500, 520, 540,
                                   560, 580, 600, 650, 700, 750,
                                   800, 850, 900, 1100, 1300, 1500),
    parametersTurnOnSumEt = cms.vdouble(    0,  100,  200,  300,  400,  500,  600,  700,  800,  900,
                                         1000, 1100, 1200, 1300, 1400, 1500
                                       )
)
