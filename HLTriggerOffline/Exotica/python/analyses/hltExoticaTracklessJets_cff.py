import FWCore.ParameterSet.Config as cms

TracklessJetsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_SingleCentralPFJet170_CFMax0p1_v",
        "HLT_DiCentralPFJet170_CFMax0p1_v",
        "HLT_DiCentralPFJet220_CFMax0p3_v",
        "HLT_DiCentralPFJet330_CFMax0p5_v",
        "HLT_DiCentralPFJet170_v",
        "HLT_DiCentralPFJet430_v"
        ),
    recPFJetLabel   = cms.InputTag("ak4PFJets"),
    recCaloJetLabel = cms.InputTag("ak4CaloJets"),

    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                  ),

    dropPt3 = cms.bool(True)
)
