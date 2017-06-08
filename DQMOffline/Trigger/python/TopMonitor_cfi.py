import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltTOPmonitoring = topMonitoring.clone()
hltTOPmonitoring.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/default/')
hltTOPmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.int32 (  30   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltTOPmonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.int32 (  60   ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  300  ),
)
hltTOPmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.int32 (  32  ),
  xmin  = cms.double( -3.2 ),
  xmax  = cms.double(  3.2 ),
)
hltTOPmonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.int32 (  24  ),
  xmin  = cms.double( -2.4 ),
  xmax  = cms.double(  2.4 ),
)
hltTOPmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.int32 (   60  ),
  xmin  = cms.double(   0   ),
  xmax  = cms.double(  600  ),
)

hltTOPmonitoring.met       = cms.InputTag("pfMetEI") # pfMet
hltTOPmonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltTOPmonitoring.electrons = cms.InputTag("gedGsfElectrons") # while pfIsolatedElectronsEI are reco::PFCandidate !
hltTOPmonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !

hltTOPmonitoring.HTdefinition = cms.string('pt>30 & abs(eta)<2.5')
hltTOPmonitoring.leptJetDeltaRmin = cms.double(0.4)

hltTOPmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltTOPmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltTOPmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltTOPmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltTOPmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltTOPmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltTOPmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltTOPmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltTOPmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltTOPmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltTOPmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltTOPmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

