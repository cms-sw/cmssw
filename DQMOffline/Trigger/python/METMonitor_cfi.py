import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.metMonitoring_cfi import metMonitoring

hltMETmonitoring = metMonitoring.clone(
    FolderName = 'HLT/JME/MET/PFMETNoMu120/',
    met       = "pfMet", # pfMet
    jets      = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
    electrons = "gedGsfElectrons", # while pfIsolatedElectronsEI are reco::PFCandidate !
    muons     = "muons", # while pfIsolatedMuonsEI are reco::PFCandidate !

    histoPSet = dict(lsPSet = dict(
                         nbins =  250,
                         xmin  =   0.,
                         xmax  =  2500.),
                    metPSet = dict(
                         nbins = 200,
                         xmin  = -0.5,
                         xmax  = 19999.5)
                    ),

    numGenericTriggerEventPSet = dict(
        andOr         = False,
        #dbLabel       = "ExoDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
        andOrHlt      = True, # True:=OR; False:=AND
        hltInputTag   =  "TriggerResults::HLT",
        hltPaths      = ["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*"], # HLT_ZeroBias_v
        #hltDBKey      = "EXO_HLT_MET",
        errorReplyHlt =  False,
        verbosityLevel = 0),

    denGenericTriggerEventPSet = dict(
        andOr         =  False,
        dcsInputTag   = "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
        andOrDcs      = False, 
        errorReplyDcs = True, 
        verbosityLevel = 1,
        hltPaths      = ["HLT_IsoMu27_v*"])
       # hltPaths      = [])
)

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(hltMETmonitoring,
                         numGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(True)),
                         denGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(True)))


