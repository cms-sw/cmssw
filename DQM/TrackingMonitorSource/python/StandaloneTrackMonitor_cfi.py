import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

from DQM.TrackingMonitorSource.standaloneTrackMonitorDefault_cfi import standaloneTrackMonitorDefault
standaloneTrackMonitor = standaloneTrackMonitorDefault.clone(
    moduleName        = "StandaloneTrackMonitor",
    folderName        = "highPurityTracks",
    vertexTag         = "selectedPrimaryVertices", # "offlinePrimaryVertices",
    puTag             = "addPileupInfo",
    clusterTag        = "siStripClusters",
    AlgoName          = 'GenTk', # 'generalTracks',
    trackInputTag     = 'selectedTracks',
    offlineBeamSpot   = 'offlineBeamSpot',
    trackQuality      = 'highPurity',
    TCProducer        = "initialStepTrackCandidates",
    MVAProducers      = ["initialStepClassifier1", "initialStepClassifier2"],
    TrackProducerForMVA = "initialStepTracks",
    doPUCorrection    = False,
    isMC              = False,
    puScaleFactorFile = "PileupScaleFactor_282037_ZtoMM.root",
    trackScaleFactorFile = "PileupScaleFactor_282037_ZtoMM.root",
    haveAllHistograms = True,
    verbose           = False,
    trackEtaH         = dict(Xbins = 60,  Xmin = -3.0, Xmax = 3.0),
    trackPtH          = dict(Xbins = 100, Xmin =  0.0 ,Xmax = 100.0),
    trackPhiH         = dict(Xbins = 100, Xmin = 3.15, Xmax = 3.15),
    #trackMVAH        = dict(Xbins = 100 ,Xmin = -1.0, Xmax = 1.0)
)
