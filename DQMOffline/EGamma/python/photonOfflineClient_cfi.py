import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.EGamma.photonAnalyzer_cfi import *


photonOfflineClient = DQMEDHarvester("PhotonOfflineClient",

    ComponentName = cms.string('photonOfflineClient'),
    analyzerName = cms.string('gedPhotonAnalyzer'),
    standAlone = cms.bool(False),
    batch = cms.bool(False),                                     


    cutStep = photonAnalysis.cutStep,
    numberOfSteps = photonAnalysis.numberOfSteps,
    minimalSetOfHistos = photonAnalysis.minimalSetOfHistos,                                
    excludeBkgHistos = photonAnalysis.excludeBkgHistos,

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
                                     
    OutputFileName = cms.string('DQMOfflinePhotonsAfterSecondStep.root'),
)
