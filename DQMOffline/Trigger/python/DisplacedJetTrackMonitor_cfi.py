import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DisplacedJetTrackMonitoring_cfi import DisplacedJetTrackMonitoring

hltDJTrackmonitoring = DisplacedJetTrackMonitoring.clone()
hltDJTrackmonitoring.FolderName = cms.string('HLT/DisplacedJet/')
hltDJTrackmonitoring.histoPSet.ntrackPSet = cms.PSet(
    nbins = cms.int32(31),
    xmin = cms.double(-0.5),
    xmax = cms.double(30.5),

)

hltDJTrackmonitoring.calojets = cms.InputTag("ak4CaloJets")
hltDJTrackmonitoring.tracks  = cms.InputTag("generalTracks")
hltDJTrackmonitoring.calojetSelection = cms.string("pt > 40 && eta <2.0")
hltDJTrackmonitoring.trackSelection = cms.string("pt > 1.0 && highPurity")

hltDJTrackmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltDJTrackmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltDJTrackmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltDJTrackmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltDJTrackmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*") # HLT_ZeroBias_v*
#hltDJTrackmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("EXO_HLT_HT")
hltDJTrackmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltDJTrackmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltDJTrackmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltDJTrackmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltDJTrackmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltDJTrackmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltDJTrackmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltDJTrackmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)
hltDJTrackmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_IsoMu27_v*","HLT_IsoTkMu27_v*");
