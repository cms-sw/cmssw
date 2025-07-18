import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.bphMonitoring_cfi import bphMonitoring as _bphMonitoring

hltBPHmonitoring = _bphMonitoring.clone(
    FolderName = 'HLT/BPH/Dimuon_10_Jpsi_Barrel/',
    tnp = True,
    max_dR = 1.4,
    minmass = 2.596,
    maxmass = 3.596,
    Upsilon = 0,
    Jpsi = 0,
    seagull = 0,
    ptCut = 0,
    displaced = 0,

#hltBPHmonitoring.options = cms.untracked.PSet(
#    SkipEvent = cms.untracked.vstring('ProductNotFound')
#)

    histoPSet = dict(
        ptBinning = [-0.5, 0, 2, 4, 8, 10, 12, 16, 20, 25, 30, 35, 40, 50],
        dMuPtBinning = [6, 8, 12, 16,  20,  25, 30, 35, 40, 50, 70],
        phiPSet = dict(
            nbins = 8,
            xmin  = -3.2,
            xmax  = 3.2,
        ),
        etaPSet = dict(
            nbins = 12,
            xmin  = -2.4,
            xmax  = 2.4,
        ),
        d0PSet = dict(
            nbins = 50,
            xmin  = -5.,
            xmax  = 5,
        ),
        z0PSet = dict(
            nbins = 60,
            xmin  = -15,
            xmax  = 15,
        ),

        dRPSet = dict(
            nbins = 26,
            xmin  = 0,
            xmax  = 1.3,
        ),

        massPSet = dict(
            nbins = 140,
            xmin  = 0,
            xmax  = 7.,
        ),
        BmassPSet = dict(
            nbins = 20,
            xmin  = 5.1,
            xmax  = 5.5,
        ),

        dcaPSet = dict(
            nbins = 10,
            xmin  = 0,
            xmax  = 0.5,
        ),

        dsPSet = dict(
            nbins = 15,
            xmin  = 0,
            xmax  = 60,
        ),

        cosPSet = dict(
            nbins = 10,
            xmin  = 0.9,
            xmax  = 1,
        ),
    
        probBinning = [0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    ),
    tracks = "generalTracks", # tracks??
    offlinePVs = "offlinePrimaryVertices", # PVs
    beamSpot = "offlineBeamSpot", # 
    muons = "muons", # 
    photons = "photons", # 
    hltTriggerSummaryAOD   = "hltTriggerSummaryAOD::HLT",
    #DMSelection_ref = "",
    #muoSelection_ref = "",
    #muoSelection_ = "",

    numGenericTriggerEventPSet = dict(
        andOr         =  False,
        #dbLabel       = "BPHDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
        andOrHlt      = True, # True:=OR; False:=AND
        andOrL1      = True, # True:=OR; False:=AND
        hltInputTag   = "TriggerResults::HLT",
        hltPaths      = ["HLT_Dimuon0_Jpsi_L1_NoOS_v*"], # HLT_ZeroBias_v*
        #l1Algorithms      = ["L1_DoubleMu0_SQ"], # HLT_ZeroBias_v*
        #hltDBKey      = "diMu10",
        errorReplyHlt = False,
        errorReplyL1 =  True,
        l1BeforeMask = True, 
        verbosityLevel = 0
    ),
    denGenericTriggerEventPSet = dict(
        andOr         = False,
        andOrHlt      = True,# True:=OR; False:=AND
        #dcsInputTag   =  "scalersRawToDigi",
        #dcsRecordInputTag = "onlineMetaDataDigis",
        hltInputTag   =  "TriggerResults::HLT" ,
        hltPaths  = ["HLT_Mu7p5_Track2_Jpsi_v*" ], #reference
        #l1Algorithms      = ["L1_DoubleMu0_SQ"], # HLT_ZeroBias_v*
        #dcsPartitions = [0,1,2,3,5,6,7,8,9,12,13,14,15,16,17,20,22,24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel
        andOrDcs      = False,
        errorReplyDcs =  True,
        verbosityLevel = 0,
    )
)

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(hltBPHmonitoring,
                         stageL1Trigger = 2,
                         numGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(False)),
                         denGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(False))
                         )

