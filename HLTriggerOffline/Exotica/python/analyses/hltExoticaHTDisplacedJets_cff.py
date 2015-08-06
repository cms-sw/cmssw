import FWCore.ParameterSet.Config as cms

HTDisplacedJetsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_HT275_v",
        "HLT_HT425_v",
        "HLT_HT575_v",

        "HLT_HT650_DisplacedDijet80_Inclusive_v",
        "HLT_HT750_DisplacedDijet80_Inclusive_v",
        "HLT_HT350_DisplacedDijet80_DisplacedTrack_v",
        "HLT_HT350_DisplacedDijet80_Tight_DisplacedTrack_v",
        # 5e33, 7e33 menus
        "HLT_HT500_DisplacedDijet40_Inclusive_v",
        "HLT_HT550_DisplacedDijet40_Inclusive_v",
        "HLT_HT350_DisplacedDijet40_DisplacedTrack_v",
        #"HLT_HT350_DisplacedDijet80_DisplacedTrack_v",
        "HLT_VBF_DisplacedJet40_DisplacedTrack_v",
        "HLT_VBF_DisplacedJet40_Hadronic_v",
        "HLT_VBF_DisplacedJet40_TightID_DisplacedTrack_v",
        "HLT_VBF_DisplacedJet40_TightID_Hadronic_v",
        # 1.4e34 menus
        "HLT_VBF_DisplacedJet40_VTightID_Hadronic_v",
        "HLT_VBF_DisplacedJet40_VVTightID_DisplacedTrack_v",
        "HLT_VBF_DisplacedJet40_VVTightID_Hadronic_v",
        "HLT_VBF_DisplacedJet40_VTightID_DisplacedTrack_v"
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
                                   320, 340, 360, 380, 400,
                                   420, 440, 460, 480, 500,
                                   520, 540, 560, 580, 600,
                                   620, 640, 660, 680, 700,
                                   750, 800, 850, 900, 950, 1000,
                                   1100, 1200, 1300, 1400, 1500)
)
