import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.htMonitoring_cfi import htMonitoring

# config file for monitoring the trigger efficiency vs softdropmass of the leading jet
# see python/HTMonitor_cfi.py or plugins/HTMonitor.h or plugins/HTMonitor.cc for more details

hltSoftdropmonitoring = htMonitoring.clone()
hltSoftdropmonitoring.FolderName = cms.string('HLT/HT/PFMETNoMu120/')
hltSoftdropmonitoring.quantity = cms.string('softdrop') # set quantity to leading jet softdropmass
hltSoftdropmonitoring.jetSelection = cms.string("pt > 65 && eta < 2.4")
hltSoftdropmonitoring.dEtaCut     = cms.double(1.3)
hltSoftdropmonitoring.histoPSet.htBinning = cms.vdouble (0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90., 95., 100., 105., 110., 115., 120., 125., 130., 135., 140., 145., 150., 155., 160., 165., 170., 175., 180., 185., 190., 195., 200.)
hltSoftdropmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32 (  200  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(19999.5),
)
hltSoftdropmonitoring.met       = cms.InputTag("pfMetEI")
hltSoftdropmonitoring.jets      = cms.InputTag("ak8PFJetsPuppiSoftDrop") # dont set this to non-SoftdropJets
hltSoftdropmonitoring.electrons = cms.InputTag("gedGsfElectrons")
hltSoftdropmonitoring.muons     = cms.InputTag("muons")

hltSoftdropmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )

hltSoftdropmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltSoftdropmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltSoftdropmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")
hltSoftdropmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltSoftdropmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltSoftdropmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltSoftdropmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltSoftdropmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29
hltSoftdropmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltSoftdropmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltSoftdropmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)
hltSoftdropmonitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_IsoMu27_v*")
