import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.jetmetMonitoring_cfi import jetmetMonitoring
hltJetMETmonitoring = jetmetMonitoring.clone()
hltJetMETmonitoring.FolderName = cms.string('HLT/JetMonitor/PFJet450/')
hltJetMETmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.int32 (  100  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(999.5),
)
hltJetMETmonitoring.met       = cms.InputTag("pfMet") # pfMet
hltJetMETmonitoring.pfjets      = cms.InputTag("ak4PFJetsCHS") # ak4PFJets, ak4PFJetsCHS
hltJetMETmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltJetMETmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !
hltJetMETmonitoring.ptcut     = cms.double(20) # while pfIsolatedMuonsEI are reco::PFCandidate !
hltJetMETmonitoring.ispfjettrg = cms.bool(True) # is PFJet Trigge  ?
hltJetMETmonitoring.iscalojettrg = cms.bool(False) # is CaloJet Trigge  ?
hltJetMETmonitoring.ismettrg = cms.bool(False) # is PFJet Trigge  ?

hltJetMETmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltJetMETmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("JetMETDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltJetMETmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltJetMETmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltJetMETmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFJet450_v*") # HLT_ZeroBias_v*
hltJetMETmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltJetMETmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltJetMETmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltJetMETmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltJetMETmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltJetMETmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltJetMETmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltJetMETmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

