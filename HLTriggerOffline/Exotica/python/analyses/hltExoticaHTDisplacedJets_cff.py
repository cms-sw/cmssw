import FWCore.ParameterSet.Config as cms

HTDisplacedJetsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_HT425_v", # Claimed path for Run3
        #2017 
        "HLT_HT430_DisplacedDijet40_DisplacedTrack_v", # Claimed path for Run3
#        "HLT_HT430_DisplacedDijet60_DisplacedTrack_v", # Claimed path for Run3, but a backup so no need to monitor it closely here
        "HLT_HT650_DisplacedDijet60_Inclusive_v", # Claimed path for Run3
#        "HLT_HT400_DisplacedDijet40_DisplacedTrack_v", # Claimed path for Run3, but a control path so no need to monitor it closely here
#        "HLT_HT550_DisplacedDijet60_Inclusive_v" # Claimed path for Run3, but a control path so no need to monitor it closely here
        "HLT_HT430_DisplacedDijet30_Inclusive1PtrkShortSig5_v", # New path for Run 3
        "HLT_Mu6HT240_DisplacedDijet30_Inclusive1PtrkShortSig5_DisplacedLoose_v", # New path for Run 3
        "HLT_HT430_DelayedJet40_SingleDelay1nsTrackless_v", # New path for Run 3
        "HLT_HT430_DelayedJet40_SingleDelay2nsInclusive_v", # New path for Run 3
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
