import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.metPlusTrackMonitoring_cfi import metPlusTrackMonitoring

hltMETplusTrackMonitoring = metPlusTrackMonitoring.clone()
hltMETplusTrackMonitoring.FolderName = cms.string('HLT/MET/MET105_IsoTrk50/')
hltMETplusTrackMonitoring.histoPSet.lsPSet = cms.PSet(
  nbins = cms.uint32(  250 ),
  xmin  = cms.double(    0.),
  xmax  = cms.double( 2500.),
)
hltMETplusTrackMonitoring.histoPSet.metPSet = cms.PSet(
  nbins = cms.uint32 (100),
  xmin  = cms.double(-0.5),
  xmax  = cms.double(999.5),
)
hltMETplusTrackMonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32 (100),
  xmin  = cms.double(-0.5),
  xmax  = cms.double(999.5),
)
hltMETplusTrackMonitoring.histoPSet.etaPSet = cms.PSet(
  nbins = cms.uint32 (24),
  xmin  = cms.double(-2.4),
  xmax  = cms.double(2.4),
)
hltMETplusTrackMonitoring.histoPSet.phiPSet = cms.PSet(
  nbins = cms.uint32(32),
  xmin  = cms.double(-3.2),
  xmax  = cms.double(3.2),
)

# Define 100 logarithmic bins from 10^0 to 10^3 GeV
binsLogX_METplusTrack = []
nBinsLogX_METplusTrack = 100
powerLo_METplusTrack = 0.0
powerHi_METplusTrack = 3.0
binPowerWidth_METplusTrack = (powerHi_METplusTrack - powerLo_METplusTrack) / nBinsLogX_METplusTrack
for ibin in range(nBinsLogX_METplusTrack + 1):
    binsLogX_METplusTrack.append( pow(10, powerLo_METplusTrack + ibin * binPowerWidth_METplusTrack) )

hltMETplusTrackMonitoring.histoPSet.metBinning = cms.vdouble(binsLogX_METplusTrack)
hltMETplusTrackMonitoring.histoPSet.ptBinning = cms.vdouble(binsLogX_METplusTrack)

hltMETplusTrackMonitoring.met       = cms.InputTag("caloMet") # caloMet
hltMETplusTrackMonitoring.jets      = cms.InputTag("pfJetsEI") # ak4PFJets, ak4PFJetsCHS
hltMETplusTrackMonitoring.muons     = cms.InputTag("muons") # while pfIsolatedMuonsEI are reco::PFCandidate !

hltMETplusTrackMonitoring.muonSelection = cms.string('pt>26 && abs(eta)<2.1 && (pfIsolationR04.sumChargedHadronPt+pfIsolationR04.sumPhotonEt+pfIsolationR04.sumNeutralHadronEt-0.5*pfIsolationR04.sumPUPt)/pt<0.12')
hltMETplusTrackMonitoring.vtxSelection = cms.string('ndof>=4 && abs(z)<24.0 && position.Rho<2.0')
hltMETplusTrackMonitoring.nmuons = cms.uint32(1)
hltMETplusTrackMonitoring.leadJetEtaCut = cms.double(2.4)

hltMETplusTrackMonitoring.numGenericTriggerEventPSet.andOr          = cms.bool( False )
#hltMETplusTrackMonitoring.numGenericTriggerEventPSet.dbLabel        = cms.string("ExoDQMTrigger") # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !
hltMETplusTrackMonitoring.numGenericTriggerEventPSet.andOrHlt       = cms.bool(True)# True:=OR; False:=AND
hltMETplusTrackMonitoring.numGenericTriggerEventPSet.hltInputTag    = cms.InputTag( "TriggerResults::HLT" )
hltMETplusTrackMonitoring.numGenericTriggerEventPSet.hltPaths       = cms.vstring("HLT_MET105_IsoTrk50_v*") # HLT_ZeroBias_v
#hltMETplusTrackMonitoring.numGenericTriggerEventPSet.hltDBKey       = cms.string("EXO_HLT_MET")
hltMETplusTrackMonitoring.numGenericTriggerEventPSet.errorReplyHlt  = cms.bool( False )
hltMETplusTrackMonitoring.numGenericTriggerEventPSet.verbosityLevel = cms.uint32(0)

hltMETplusTrackMonitoring.denGenericTriggerEventPSet.andOr          = cms.bool( False )
hltMETplusTrackMonitoring.denGenericTriggerEventPSet.dcsInputTag    = cms.InputTag( "scalersRawToDigi" )
hltMETplusTrackMonitoring.denGenericTriggerEventPSet.dcsPartitions  = cms.vint32 ( 24, 25, 26, 27, 28, 29 ) # 24-27: strip, 28-29: pixel, we should add all other detectors !
hltMETplusTrackMonitoring.denGenericTriggerEventPSet.andOrDcs       = cms.bool( False )
hltMETplusTrackMonitoring.denGenericTriggerEventPSet.errorReplyDcs  = cms.bool( True )
hltMETplusTrackMonitoring.denGenericTriggerEventPSet.verbosityLevel = cms.uint32(1)
hltMETplusTrackMonitoring.denGenericTriggerEventPSet.hltPaths       = cms.vstring("HLT_IsoMu27_v*")
