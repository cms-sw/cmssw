import FWCore.ParameterSet.Config as cms

TracklessJetsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
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
