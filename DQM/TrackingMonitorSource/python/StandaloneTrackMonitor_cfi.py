import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

standaloneTrackMonitor = DQMEDAnalyzer('StandaloneTrackMonitor',
    moduleName        = cms.untracked.string("StandaloneTrackMonitor"),
    folderName        = cms.untracked.string("highPurityTracks"),
    vertexTag         = cms.untracked.InputTag("selectedPrimaryVertices"),
#    vertexTag         = cms.untracked.InputTag("offlinePrimaryVertices"),
    puTag             = cms.untracked.InputTag("addPileupInfo"),
    clusterTag        = cms.untracked.InputTag("siStripClusters"),
    AlgoName          = cms.untracked.string('GenTk'),
#    trackInputTag     = cms.untracked.InputTag('generalTracks'),
    trackInputTag     = cms.untracked.InputTag('selectedTracks'),
    offlineBeamSpot   = cms.untracked.InputTag('offlineBeamSpot'),
    pfCombinedSecondaryVertexV2BJetTags = cms.untracked.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags","","RECO"),
    trackQuality      = cms.untracked.string('highPurity'),
    TCProducer        = cms.untracked.InputTag("initialStepTrackCandidates"),
    MVAProducers      = cms.untracked.vstring("initialStepClassifier1", "initialStepClassifier2"),
    TrackProducerForMVA = cms.untracked.InputTag("initialStepTracks"),
    doPUCorrection    = cms.untracked.bool(False),
    isMC              = cms.untracked.bool(False),
    puScaleFactorFile = cms.untracked.string("PileupScaleFactor_282037_ZtoMM.root"),
    trackScaleFactorFile = cms.untracked.string("PileupScaleFactor_282037_ZtoMM.root"),
    haveAllHistograms = cms.untracked.bool(True),
    verbose           = cms.untracked.bool(False),
    trackEtaH         = cms.PSet(Xbins = cms.int32(60), Xmin = cms.double(-3.0),Xmax = cms.double(3.0)),
    trackPtH          = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(0.0),Xmax = cms.double(100.0)),
    trackMVAH          = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-1.0),Xmax = cms.double(1.0))
)
