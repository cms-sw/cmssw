import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.htMonitoring_cfi import htMonitoring

# config file for monitoring the trigger efficiency vs invariant dijetmass of the two leading jets
# see python/HTMonitor_cfi.py or plugins/HTMonitor.h or plugins/HTMonitor.cc for more details

hltMjjmonitoring = htMonitoring.clone(
    FolderName = 'HLT/HT/PFMETNoMu120/',
    quantity = 'Mjj', # set quantity to invariant dijetmass
    jetSelection = "pt > 200 && eta < 2.4",
    dEtaCut     = 1.3,
    met       = "pfMetEI",
    jets      = "ak8PFJetsPuppi",
    electrons = "gedGsfElectrons",
    muons     = "muons"
)
hltMjjmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32 (  200  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(19999.5),
)
hltMjjmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )

hltMjjmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltMjjmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltMjjmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")
hltMjjmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltMjjmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltMjjmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltMjjmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltMjjmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29
hltMjjmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltMjjmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltMjjmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)
hltMjjmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_IsoMu27_v*")
