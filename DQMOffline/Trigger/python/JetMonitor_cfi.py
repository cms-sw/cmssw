import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.jetMonitoring_cfi import jetMonitoring

hltJetMETmonitoring = jetMonitoring.clone()
hltJetMETmonitoring.FolderName = cms.string('HLT/JME/Jets/AK4/PFJet/HLT_PFJet450/')
hltJetMETmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32(250),
  xmin  = cms.double(0.),
  xmax  = cms.double(2500.),
)
hltJetMETmonitoring.histoPSet.jetPSet = cms.PSet(
  nbins = cms.uint32(200),
  xmin  = cms.double(-0.5),
  xmax  = cms.double(999.5),
)
hltJetMETmonitoring.histoPSet.jetPtThrPSet = cms.PSet(
  nbins = cms.uint32(180),
  xmin  = cms.double(0.),
  xmax  = cms.double(900),
)
hltJetMETmonitoring.jetSrc = 'ak4PFJets' # ak4PFJets, ak4PFJetsCHS
hltJetMETmonitoring.ptcut = 20.
hltJetMETmonitoring.ispfjettrg = True # is PFJet Trigger ?
hltJetMETmonitoring.iscalojettrg = False # is CaloJet Trigger ?

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

