import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.dijetMonitoring_cfi import dijetMonitoring
DiPFjetAve40_Prommonitoring = dijetMonitoring.clone()
DiPFjetAve40_Prommonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve40/')
DiPFjetAve40_Prommonitoring.histoPSet.dijetPSet = cms.PSet(
  nbins = cms.uint32 (  200  ),
  xmin  = cms.double(   0),
  xmax  = cms.double(1000.),
)
DiPFjetAve40_Prommonitoring.histoPSet.dijetPtThrPSet = cms.PSet(
  nbins = cms.uint32 (  50 ),
  xmin  = cms.double(   0.),
  xmax  = cms.double(100.),
)
DiPFjetAve40_Prommonitoring.met       = cms.InputTag("pfMetEI") # pfMet
#DiPFjetAve40_Prommonitoring.pfjets    = cms.InputTag("ak4PFJets") # ak4PFJets, ak4PFJetsCHS
DiPFjetAve40_Prommonitoring.dijetSrc  = cms.InputTag("ak4PFJets") # ak4PFJets, ak4PFJetsCHS
DiPFjetAve40_Prommonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
DiPFjetAve40_Prommonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !
DiPFjetAve40_Prommonitoring.ptcut     = cms.double(20) # while pfIsolatedMuonsEI are reco::PFCandidate !

DiPFjetAve40_Prommonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
DiPFjetAve40_Prommonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("JetMETDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
DiPFjetAve40_Prommonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
DiPFjetAve40_Prommonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
DiPFjetAve40_Prommonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_DiPFJetAve40_v*") # HLT_ZeroBias_v*
DiPFjetAve40_Prommonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
DiPFjetAve40_Prommonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

DiPFjetAve40_Prommonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
DiPFjetAve40_Prommonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
DiPFjetAve40_Prommonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
DiPFjetAve40_Prommonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
DiPFjetAve40_Prommonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
DiPFjetAve40_Prommonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

