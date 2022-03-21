import FWCore.ParameterSet.Config as cms

DSTJetsPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        # For backward compatibility
#        "DST_HT250_CaloScouting_v", # Not claimed path for Run 3
        ),
    recPFMHTLabel  = cms.InputTag("recoExoticaValidationHT"),
    recPFJetLabel  = cms.InputTag("ak4PFJets"),
    recCaloMHTLabel  = cms.InputTag("recoExoticaValidationCaloHT"),
    recCaloJetLabel  = cms.InputTag("ak4CaloJets"),
    # -- Analysis specific cuts
    #MET_genCut      = cms.string("sumEt > 75"),
    #MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   )
)
