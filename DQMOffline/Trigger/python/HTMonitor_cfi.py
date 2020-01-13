import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.htMonitoring_cfi import htMonitoring

hltHTmonitoring = htMonitoring.clone()
hltHTmonitoring.FolderName = cms.string('HLT/HT/PFMETNoMu120/')
hltHTmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32(  250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltHTmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32 (  200  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(19999.5),
)
hltHTmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltHTmonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltHTmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltHTmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !

hltHTmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltHTmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltHTmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltHTmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltHTmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*") # HLT_ZeroBias_v*
#hltHTmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("EXO_HLT_HT")
hltHTmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltHTmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltHTmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltHTmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltHTmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltHTmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltHTmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltHTmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)
hltHTmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_IsoMu27_v*")
