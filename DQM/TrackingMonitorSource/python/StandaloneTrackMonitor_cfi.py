import FWCore.ParameterSet.Config as cms
standaloneTrackMonitor = cms.EDAnalyzer('StandaloneTrackMonitor',
    moduleName        = cms.untracked.string("StandaloneTrackMonitor"),
    folderName        = cms.untracked.string("highPurityTracks"),
    vertexTag         = cms.untracked.InputTag("selectedPrimaryVertices"),
    puTag             = cms.untracked.InputTag("addPileupInfo"),
    clusterTag        = cms.untracked.InputTag("siStripClusters"),
    trackInputTag     = cms.untracked.InputTag('selectedTracks'),
    offlineBeamSpot   = cms.untracked.InputTag('offlineBeamSpot'),
    trackQuality      = cms.untracked.string('highPurity'),
    doPUCorrection    = cms.untracked.bool(False),
    isMC              = cms.untracked.bool(True),
    puScaleFactorFile = cms.untracked.string("PileupScaleFactor_run203002.root"),
    haveAllHistograms = cms.untracked.bool(False),
    verbose           = cms.untracked.bool(False),
    trackEtaH         = cms.PSet(Xbins = cms.int32(60), Xmin = cms.double(-3.0),Xmax = cms.double(3.0)),
    trackPtH          = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0))
)
