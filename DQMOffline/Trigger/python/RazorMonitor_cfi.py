import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.razorMonitoring_cfi import razorMonitoring

hltRazorMonitoring = razorMonitoring.clone()
hltRazorMonitoring.FolderName = cms.string('HLT/SUSY/RsqMR270_Rsq0p09_MR200')

hltRazorMonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltRazorMonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS

hltRazorMonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltRazorMonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True) # True:=OR; False:=AND
hltRazorMonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltRazorMonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_RsqMR300_Rsq0p09_MR200_v*") 
hltRazorMonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltRazorMonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltRazorMonitoring.denGenericTriggerEventPSet.andOr          = cms.bool( False )
hltRazorMonitoring.denGenericTriggerEventPSet.andOrHlt       = cms.bool( True )
hltRazorMonitoring.denGenericTriggerEventPSet.dcsInputTag    = cms.InputTag( "scalersRawToDigi" )
hltRazorMonitoring.denGenericTriggerEventPSet.dcsPartitions  = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltRazorMonitoring.denGenericTriggerEventPSet.andOrDcs       = cms.bool( False )
hltRazorMonitoring.denGenericTriggerEventPSet.errorReplyDcs  = cms.bool( True )
hltRazorMonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)
hltRazorMonitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Ele27_WPTight_Gsf*", "HLT_Ele30_WPTight_Gsf*", "HLT_Ele32_WPTight_Gsf*")
