import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DisplacedJetHTMonitoring_cfi import DisplacedJetHTMonitoring

hltDJHTmonitoring = DisplacedJetHTMonitoring.clone()
hltDJHTmonitoring.FolderName = cms.string('HLT/DisplacedJet/')
hltDJHTmonitoring.histoPSet.calohtPSet = cms.PSet(
    nbins = cms.int32( 50 ),
    xmin  = cms.double (350.0),
    xmax  = cms.double (800.0),
)

hltDJHTmonitoring.calojets = cms.InputTag("ak4CaloJets")
hltDJHTmonitoring.calojetSelection = cms.string("pt > 40 && eta <2.0")
hltDJHTmonitoring.ncalojets = cms.int32(2)

hltDJHTmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
hltDJHTmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)
hltDJHTmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltDJHTmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*") 
hltDJHTmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltDJHTmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltDJHTmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltDJHTmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltDJHTmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 )
hltDJHTmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltDJHTmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltDJHTmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)
hltDJHTmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_IsoMu27_v*","HLT_IsoTkMu27_v*");

