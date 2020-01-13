import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.NoBPTXMonitoring_cfi import NoBPTXMonitoring

hltNoBPTXmonitoring = NoBPTXMonitoring.clone()
hltNoBPTXmonitoring.FolderName = cms.string('HLT/EXO/NoBPTX/JetE60/')
hltNoBPTXmonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32(250),
  xmin  = cms.double(0.),
  xmax  = cms.double(2500.),
)
hltNoBPTXmonitoring.histoPSet.jetEPSet = cms.PSet(
    nbins = cms.uint32(100),
    xmin  = cms.double(-0.5),
    xmax  = cms.double(999.5),
)
hltNoBPTXmonitoring.histoPSet.jetEtaPSet = cms.PSet(
    nbins = cms.uint32(100),
    xmin  = cms.double(-5.),
    xmax  = cms.double(5.),
)
hltNoBPTXmonitoring.histoPSet.jetPhiPSet = cms.PSet(
    nbins = cms.uint32(64),
    xmin  = cms.double(-3.2),
    xmax  = cms.double(3.2),
)
hltNoBPTXmonitoring.histoPSet.muonPtPSet = cms.PSet(
    nbins = cms.uint32(100),
    xmin  = cms.double(-0.5),
    xmax  = cms.double(999.5),
)
hltNoBPTXmonitoring.histoPSet.muonEtaPSet = cms.PSet(
    nbins = cms.uint32(100),
    xmin  = cms.double(-5.),
    xmax  = cms.double(5.),
)
hltNoBPTXmonitoring.histoPSet.muonPhiPSet = cms.PSet(
    nbins = cms.uint32(64),
    xmin  = cms.double(-3.2),
    xmax  = cms.double(3.2),
)
hltNoBPTXmonitoring.histoPSet.bxPSet = cms.PSet(
    nbins = cms.uint32(1800),
)
hltNoBPTXmonitoring.jets = cms.InputTag("ak4CaloJets")
hltNoBPTXmonitoring.muons = cms.InputTag("displacedStandAloneMuons")

hltNoBPTXmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltNoBPTXmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !                                                                                                           
hltNoBPTXmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND 
hltNoBPTXmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltNoBPTXmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_UncorrectedJetE60_NoBPTX3BX_v*") # HLT_ZeroBias_v*
#hltNoBPTXmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("EXO_HLT_NoBPTX") 
hltNoBPTXmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltNoBPTXmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltNoBPTXmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltNoBPTXmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltNoBPTXmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltNoBPTXmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltNoBPTXmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltNoBPTXmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltNoBPTXmonitoring.muonSelection = cms.string("hitPattern.dtStationsWithValidHits > 3 & hitPattern.numberOfValidMuonRPCHits > 1 & hitPattern.numberOfValidMuonCSCHits < 1")
hltNoBPTXmonitoring.jetSelection = cms.string("abs(eta) < 1.")
