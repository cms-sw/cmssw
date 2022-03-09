import FWCore.ParameterSet.Config as cms

CaloHTPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
#        "HLT_ECALHT800_v", # Run2 7e33 # Not claimed path for Run3
        ),
    recCaloMHTLabel  = cms.InputTag("recoExoticaValidationCaloHT"),
    recCaloJetLabel  = cms.InputTag("ak4CaloJets"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   )
)
