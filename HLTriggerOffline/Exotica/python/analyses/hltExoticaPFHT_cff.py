import FWCore.ParameterSet.Config as cms

PFHTPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
#        "HLT_PFHT350MinPFJet15_v" #2017 # Not claimed path for Run3
        "HLT_PFHT1050_v", # Claimed path for Run3
        "HLT_PFHT250_v" # Claimed path for Run3
        ),
    recPFMHTLabel  = cms.InputTag("recoExoticaValidationHT"),
    recPFJetLabel  = cms.InputTag("ak4PFJets"),
    # -- Analysis specific cuts
    #MET_genCut      = cms.string("sumEt > 75"),
    #MET_recCut      = cms.string("sumEt > 75"),
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 470, 
                                   500, 550, 600, 650, 700, 800, 900, 1000
                                   ),
    parametersTurnOnSumEt = cms.vdouble(    0,  100,  200,  300,  400,  500,  600,  700,  800,  900,
                                         1000, 1100, 1200, 1300, 1400, 1600, 1800, 2000
                                       )
)
