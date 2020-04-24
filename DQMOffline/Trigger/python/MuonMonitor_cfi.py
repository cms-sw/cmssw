import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.muonMonitoring_cfi import muonMonitoring

hltMuonmonitoring = muonMonitoring.clone()
hltMuonmonitoring.FolderName = cms.string('HLT/Muon/TrkMu16_DoubleTrkMu6NoFiltersNoVtx/')
hltMuonmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltMuonmonitoring.histoPSet.muonPSet = cms.PSet(
  nbins = cms.uint32(  500  ), ### THIS SHOULD BE VARIABLE BINNING !!!!!
  xmin  = cms.double(  0.0),
  xmax  = cms.double(500),
)
hltMuonmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltMuonmonitoring.muons = cms.InputTag("muons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltMuonmonitoring.nmuons = cms.uint32(0)


hltMuonmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltMuonmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltMuonmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltMuonmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltMuonmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx_v*") # HLT_ZeroBias_v*
#hltMuonmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("EXO_HLT_MET")
hltMuonmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltMuonmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltMuonmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltMuonmonitoring.denGenericTriggerEventPSet.andOrHlt        = cms.bool( True )
hltMuonmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltMuonmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("") # HLT_ZeroBias_v*
hltMuonmonitoring.denGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltMuonmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltMuonmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltMuonmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltMuonmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltMuonmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

