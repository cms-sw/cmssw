import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.photonMonitoring_cfi import photonMonitoring

hltPhotonmonitoring = photonMonitoring.clone()
hltPhotonmonitoring.FolderName = cms.string('HLT/Photon/Photon200/')
hltPhotonmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32 ( 250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltPhotonmonitoring.histoPSet.photonPSet = cms.PSet(
  nbins = cms.uint32(  500  ),
  xmin  = cms.double(  0.0),
  xmax  = cms.double(5000),
)
hltPhotonmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltPhotonmonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltPhotonmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltPhotonmonitoring.photons = cms.InputTag("gedPhotons") # while pfIsolatedElectronsEI are reco::PFCandidate !

hltPhotonmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltPhotonmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltPhotonmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltPhotonmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltPhotonmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_Photon175_v*") # HLT_ZeroBias_v*
#hltPhotonmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("EXO_HLT_MET")
hltPhotonmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltPhotonmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltPhotonmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltPhotonmonitoring.denGenericTriggerEventPSet.andOrHlt        = cms.bool( True )
hltPhotonmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltPhotonmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFJet40_v*","HLT_PFJet60_v*","HLT_PFJet80_v*") # HLT_ZeroBias_v*
hltPhotonmonitoring.denGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltPhotonmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltPhotonmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltPhotonmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltPhotonmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltPhotonmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)
