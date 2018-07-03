import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.objMonitoring_cfi import objMonitoring

hltobjmonitoring = objMonitoring.clone()
hltobjmonitoring.FolderName = cms.string('HLT/GENERIC/')
hltobjmonitoring.doMETHistos = cms.bool(True)
hltobjmonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32 (  200  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(19999.5),
)
hltobjmonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32 (  64  ),
  xmin  = cms.double(   -3.1416),
  xmax  = cms.double(3.1416),
)
hltobjmonitoring.doJetHistos = cms.bool(True)
hltobjmonitoring.histoPSet.jetetaPSet = cms.PSet(
  nbins = cms.uint32 (  100  ),
  xmin  = cms.double(   -5),
  xmax  = cms.double(5),
)
hltobjmonitoring.histoPSet.detajjPSet = cms.PSet(
  nbins = cms.uint32 (  90  ),
  xmin  = cms.double(   0),
  xmax  = cms.double(9),
)
hltobjmonitoring.histoPSet.dphijjPSet = cms.PSet(
  nbins = cms.uint32 (  64  ),
  xmin  = cms.double(   0),
  xmax  = cms.double(3.1416),
)
hltobjmonitoring.histoPSet.mindphijmetPSet = cms.PSet(
  nbins = cms.uint32 (  64  ),
  xmin  = cms.double(   0),
  xmax  = cms.double(3.1416),
)
hltobjmonitoring.doHTHistos = cms.bool(True)
hltobjmonitoring.histoPSet.htPSet = cms.PSet(
  nbins = cms.uint32 (  60  ),
  xmin  = cms.double(   -0.5),
  xmax  = cms.double(1499.5),
)
hltobjmonitoring.doHMesonGammaHistos = cms.bool(False)
hltobjmonitoring.histoPSet.hmgetaPSet = cms.PSet(
  nbins = cms.uint32 (  60  ),
  xmin  = cms.double(   -2.6),
  xmax  = cms.double(2.6),
)

hltobjmonitoring.met       = cms.InputTag("pfMet")
hltobjmonitoring.jets      = cms.InputTag("ak4PFJetsCHS")
hltobjmonitoring.electrons = cms.InputTag("gedGsfElectrons")
hltobjmonitoring.muons     = cms.InputTag("muons")
hltobjmonitoring.photons   = cms.InputTag("gedPhotons")
hltobjmonitoring.tracks    = cms.InputTag("generalTracks")

hltobjmonitoring.numGenericTriggerEventPSet.andOr         = cms.bool( False )
#hltobjmonitoring.numGenericTriggerEventPSet.dbLabel       = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltobjmonitoring.numGenericTriggerEventPSet.andOrHlt      = cms.bool(True)# True:=OR; False:=AND
hltobjmonitoring.numGenericTriggerEventPSet.hltInputTag   = cms.InputTag( "TriggerResults::HLT" )
hltobjmonitoring.numGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*") # HLT_ZeroBias_v*
#hltobjmonitoring.numGenericTriggerEventPSet.hltDBKey      = cms.string("EXO_HLT_MET")
hltobjmonitoring.numGenericTriggerEventPSet.errorReplyHlt = cms.bool( False )
hltobjmonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

hltobjmonitoring.denGenericTriggerEventPSet.andOr         = cms.bool( False )
hltobjmonitoring.denGenericTriggerEventPSet.dcsInputTag   = cms.InputTag( "scalersRawToDigi" )
hltobjmonitoring.denGenericTriggerEventPSet.dcsPartitions = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltobjmonitoring.denGenericTriggerEventPSet.andOrDcs      = cms.bool( False )
hltobjmonitoring.denGenericTriggerEventPSet.errorReplyDcs = cms.bool( True )
hltobjmonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)

