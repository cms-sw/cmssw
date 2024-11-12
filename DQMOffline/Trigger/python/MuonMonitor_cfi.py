import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.muonMonitoring_cfi import muonMonitoring

hltMuonmonitoring = muonMonitoring.clone(
    FolderName = 'HLT/Muon/TrkMu16_DoubleTrkMu6NoFiltersNoVtx/',
    met       = "pfMet", # pfMet
    muons = "muons", # while pfIsolatedElectronsEI are reco::PFCandidate !
    nmuons = 0,

    histoPSet = dict(
        lsPSet = dict(
                    nbins =  250 ,
                    xmin  =   0.,
                    xmax  =  2500.),
        muonPSet = dict(
                    nbins =  500 , ### THIS SHOULD BE VARIABLE BINNING !!!!!
                    xmin  =  0.0,
                    xmax  = 500),
    ),

    numGenericTriggerEventPSet = dict(
        andOr         = False,
        #dbLabel       = "ExoDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
        andOrHlt      = True,# True:=OR; False:=AND
        hltInputTag   =  "TriggerResults::HLT" ,
        hltPaths      = ["HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx_v*"], # HLT_ZeroBias_v*
        #hltDBKey      = "EXO_HLT_MET",
        errorReplyHlt =  False,
        verbosityLevel = 1),

    denGenericTriggerEventPSet = dict(
        andOr         =  False,
        andOrHlt      =  True,
        hltInputTag   = "TriggerResults::HLT",
        hltPaths      = [""], # HLT_ZeroBias_v*
        errorReplyHlt = False,
        dcsInputTag   = "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
        andOrDcs      =  False,
        errorReplyDcs =  True,
        verbosityLevel = 1)
)

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(hltMuonmonitoring,
                         numGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(False)),
                         denGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(False))
                         )


