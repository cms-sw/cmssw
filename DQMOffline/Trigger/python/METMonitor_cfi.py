import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.metMonitoring_cfi import metMonitoring

hltMETmonitoring = metMonitoring.clone()
hltMETmonitoring.FolderName = cms.string('HLT/JME/MET/PFMETNoMu120/')
hltMETmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltMETmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32 (200),
  xmin  = cms.double(-0.5),
  xmax  = cms.double(19999.5),
)

hltMETmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltMETmonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltMETmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltMETmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !

hltMETmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltMETmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltMETmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltMETmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltMETmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*") # HLT_ZeroBias_v
#hltMETmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("EXO_HLT_MET")
hltMETmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltMETmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltMETmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltMETmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltMETmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltMETmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltMETmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltMETmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)
#hltMETmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_IsoMu27_v*");
hltMETmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring();
