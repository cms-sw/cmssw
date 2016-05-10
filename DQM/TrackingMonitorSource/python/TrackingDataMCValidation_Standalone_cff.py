import FWCore.ParameterSet.Config as cms
from DQM.TrackingMonitorSource.StandaloneTrackMonitor_cfi import *

# Primary Vertex Selector
selectedPrimaryVertices = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("!isFake && ndof >= 4 && abs(z) < 24 && abs(position.Rho) < 2.0"),
    filter = cms.bool(True)
)
# Track Selector
selectedTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag('generalTracks'),
    cut = cms.string("pt > 1.0"),
    filter = cms.bool(True)
)
# HLT path selector
hltPathFilter = cms.EDFilter("HLTPathSelector",
    processName = cms.string("HLT"),
    hltPathsOfInterest = cms.vstring("HLT_ZeroBias"),
    triggerResults = cms.untracked.InputTag("TriggerResults","","HLT"),
    triggerEvent = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
)
# Z->MuMu event selector
ztoMMEventSelector = cms.EDFilter("ZtoMMEventSelector")
muonTracks = cms.EDProducer("MuonTrackProducer")
# Z->ee event selector
ztoEEEventSelector = cms.EDFilter("ZtoEEEventSelector")
electronTracks = cms.EDProducer("ElectronTrackProducer")

standaloneTrackMonitorElec = standaloneTrackMonitor.clone(
    folderName = cms.untracked.string("ElectronTracks"),
    trackInputTag = cms.untracked.InputTag('electronTracks'),
    haveAllHistograms = cms.untracked.bool(True),
    puScaleFactorFile = cms.untracked.string("PileupScaleFactor_2015D_goldenjson_newRelvalMC.root"),
    doPUCorrection    = cms.untracked.bool(True),
    isMC              = cms.untracked.bool(True)
    )

standaloneValidationElec = cms.Sequence(
    selectedTracks
    * selectedPrimaryVertices
    * ztoEEEventSelector
    * electronTracks
    * standaloneTrackMonitorElec   
    * standaloneTrackMonitor)

standaloneTrackMonitorMuon = standaloneTrackMonitor.clone(
    folderName = cms.untracked.string("MuonTracks"),
    trackInputTag = cms.untracked.InputTag('muonTracks'),
    haveAllHistograms = cms.untracked.bool(True),
    puScaleFactorFile = cms.untracked.string("PileupScaleFactor_260627_PixDynIneff.root"),
    doPUCorrection    = cms.untracked.bool(True),
    isMC              = cms.untracked.bool(True)
    )

standaloneValidationMuon = cms.Sequence(
    selectedTracks
    * selectedPrimaryVertices
    * ztoMMEventSelector
    * muonTracks
    * standaloneTrackMonitorMuon 
    * standaloneTrackMonitor)

standaloneTrackMonitorMB = standaloneTrackMonitor.clone(
    puScaleFactorFile = cms.untracked.string("PileupScaleFactor_2015D_goldenjson.root"),
    doPUCorrection    = cms.untracked.bool(True),
    isMC              = cms.untracked.bool(True)
    )

standaloneValidationMinbias = cms.Sequence(
    hltPathFilter
    * selectedTracks
    * selectedPrimaryVertices
    * standaloneTrackMonitorMB)

