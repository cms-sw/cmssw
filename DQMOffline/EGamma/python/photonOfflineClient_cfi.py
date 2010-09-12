import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonAnalyzer_cfi import *


photonOfflineClient = cms.EDAnalyzer("PhotonOfflineClient",

    Name = cms.untracked.string('photonOfflineClient'),

    standAlone = cms.bool(False),
    batch = cms.bool(False),                                     


    cutStep = photonAnalysis.cutStep,
    numberOfSteps = photonAnalysis.numberOfSteps,


    etBin = photonAnalysis.etBin,
    etMin = photonAnalysis.etMin,
    etMax = photonAnalysis.etMax,
                             
    etaBin = photonAnalysis.etaBin,
    etaMin = photonAnalysis.etaMin,
    etaMax = photonAnalysis.etaMax,

    phiBin = photonAnalysis.phiBin,
    phiMin = photonAnalysis.phiMin,
    phiMax = photonAnalysis.phiMax,
                                     
                                     
    InputFileName = cms.untracked.string("DQMOfflinePhotonsBatch.root"),
                                     
    OutputFileName = cms.string('DQMOfflinePhotonsStandAlone.root'),
)
