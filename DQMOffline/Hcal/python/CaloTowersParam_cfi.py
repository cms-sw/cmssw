import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
calotowersAnalyzer = DQMEDAnalyzer('CaloTowersAnalyzer',
    outputFile               = cms.untracked.string(''),
    CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),
    hcalselector             = cms.untracked.string('all'),
    useAllHistos             = cms.untracked.bool(False)
)

