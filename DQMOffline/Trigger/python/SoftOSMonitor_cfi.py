import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.sosMonitoring_cfi import sosMonitoring

hltSOSmonitoring = sosMonitoring.clone()
hltSOSmonitoring.FolderName = cms.string('HLT/SOS/PFMET/')
hltSOSmonitoring.histoPSet.sosPSet = cms.PSet(
  nbins = cms.int32 (  200  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(19999.5),
)
hltSOSmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltSOSmonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltSOSmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltSOSmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !

hltSOSmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltSOSmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_DoubleMu3_PFMET50_DZ_PFMHT60_v1") # HLT_ZeroBias_v*
hltSOSmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltSOSmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v7")
hltSOSmonitoring.denGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltSOSmonitoring.turn_on= cms.string("met")
hltSOSmonitoring.met_pt_cut=cms.int32(40)
hltSOSmonitoring.mu2_pt_cut=cms.int32(3)
