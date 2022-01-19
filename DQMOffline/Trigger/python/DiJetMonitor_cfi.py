import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.dijetMonitoring_cfi import dijetMonitoring
DiPFjetAve40_Prommonitoring = dijetMonitoring.clone(
    FolderName = 'HLT/JME/Jets/AK4/PF/HLT_DiPFJetAve40/',
    met       = "pfMetEI", # pfMet
    #pfjets    = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
    dijetSrc  = "ak4PFJets", # ak4PFJets, ak4PFJetsCHS
    electrons = "gedGsfElectrons", # while pfIsolatedElectronsEI are reco::PFCandidate !
    muons     = "muons", # while pfIsolatedMuonsEI are reco::PFCandidate !
    ptcut     = 20 # while pfIsolatedMuonsEI are reco::PFCandidate !
)
DiPFjetAve40_Prommonitoring.histoPSet.dijetPSet = dict(
  nbins = 200 ,
  xmin  =  0,
  xmax  = 1000.,
)
DiPFjetAve40_Prommonitoring.histoPSet.dijetPtThrPSet = dict(
  nbins = 50 ,
  xmin  =  0.,
  xmax  = 100.,
)
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

