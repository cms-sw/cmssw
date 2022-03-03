import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.razorMonitoring_cfi import razorMonitoring

hltRazorMonitoring = razorMonitoring.clone(
    FolderName = 'HLT/SUSY/RsqMR270_Rsq0p09_MR200',
    met       = "pfMet", # pfMet
    jets      = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS


    numGenericTriggerEventPSet = dict(
        andOr     =  False,
        andOrHlt      = True, # True:=OR; False:=AND
        hltInputTag   = "TriggerResults::HLT" ,
        hltPaths      = ["HLT_RsqMR300_Rsq0p09_MR200_v*"], 
        errorReplyHlt =  False,
        verbosityLevel = 1),

    denGenericTriggerEventPSet = dict(
        andOr          = False,
        andOrHlt       =  True,
        dcsInputTag    =  "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions  = [24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
        andOrDcs       =  False,
        errorReplyDcs  = True ,
        verbosityLevel = 1,
        hltPaths = ["HLT_Ele27_WPTight_Gsf*", "HLT_Ele30_WPTight_Gsf*", "HLT_Ele32_WPTight_Gsf*"])
)

