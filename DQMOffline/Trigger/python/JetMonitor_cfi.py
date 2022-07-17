import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.jetMonitoring_cfi import jetMonitoring

hltJetMETmonitoring = jetMonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PFJet/HLT_PFJet450/',

    histoPSet = dict(
        lsPSet = dict(
                nbins = 250,
                xmin  = 0.,
                xmax  = 2500.),
        jetPSet = dict(
                nbins = 200,
                xmin  = -0.5,
                xmax  = 999.5),
        jetPtThrPSet = dict(
                nbins = 180,
                xmin  = 0.,
                xmax  = 900),
    ),
    jetSrc = 'ak4PFJets', # ak4PFJets, ak4PFJetsCHS
    ptcut = 20.,
    ispfjettrg = True, # is PFJet Trigger ?
    iscalojettrg = False, # is CaloJet Trigger ?

    numGenericTriggerEventPSet = dict(
        andOr         =  False,
        dbLabel       = "JetMETDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
        andOrHlt      = True, # True:=OR; False:=AND
        hltInputTag   =  "TriggerResults::HLT",
        hltPaths      = ["HLT_PFJet450_v*"], # HLT_ZeroBias_v*
        errorReplyHlt = False,
        verbosityLevel = 1),

    denGenericTriggerEventPSet = dict(
        andOr         =  False,
        dcsInputTag   =  "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions = [ 24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
        andOrDcs      = False,
        errorReplyDcs =  True,
        verbosityLevel = 1)
)

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(hltJetMETmonitoring,
                         numGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(True)),
                         denGenericTriggerEventPSet = dict(stage2 = cms.bool(True),
                                                           l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
                                                           ReadPrescalesFromFile = cms.bool(True)))


