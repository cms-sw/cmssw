import FWCore.ParameterSet.Config as cms

METplusTrackPSet = cms.PSet(
    hltPathsToCheck = cms.vstring(
        "HLT_MET105_IsoTrk50_v", # 2017 proposal # Claimed path for Run3
#        "HLT_MET120_IsoTrk50_v"  # 2017 proposal # Claimed path for Run3, but a backup so no need to monitor it closely here
        "HLT_PFMET105_IsoTrk50_v", # New Run 3 path
        "HLT_PFMET110_PFJet100_v", # New Run 3 path
    ),
    recPFMETLabel = cms.InputTag("pfMet"),
    #recMETLabel   = cms.InputTag("hltPFMETProducer"),
    genMETLabel   = cms.InputTag("genMetTrue"),
    recMuonLabel  = cms.InputTag("muons"),
    recElecLabel  = cms.InputTag("gedGsfElectrons"),
    #recTrackLabel = cms.InputTag("generalTracks"),
    #hltMETLabel   = cms.InputTag("hltMetClean"),
    # -- Analysis specific cuts
    minCandidates = cms.uint32(1),
    # -- Analysis specific binnings
    parametersTurnOn = cms.vdouble(   0,  10,  20,  30,  40,  50,  60,  70,   80,  90,
                                    100, 110, 120, 130, 140, 150, 160, 170,  180, 190,
                                    200, 210, 220, 230, 240, 250, 260, 270,  280, 290,
                                    300, 310, 320, 330, 340, 350, 360, 370,  380, 390,
                                    400
                                  ),
    parametersTurnOnSumEt = cms.vdouble(   0,  10,  20,  30,  40,  50,  60,  70,   80,  90,
                                         100, 110, 120, 130, 140, 150, 160, 170,  180, 190,
                                         200, 210, 220, 230, 240, 250, 260, 270,  280, 290,
                                         300, 310, 320, 330, 340, 350, 360, 370,  380, 390,
                                         400
                                       ),
    dropPt2 = cms.bool(True),
    dropPt3 = cms.bool(True),
)

