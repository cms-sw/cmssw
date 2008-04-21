import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.CaloTowersRec_cff import *
from FastSimulation.ParamL3MuonProducer.ParamL3Muon_cfi import *
towerMaker.ecalInputs = cms.VInputTag(cms.InputTag("caloRecHits","EcalRecHitsEB"), cms.InputTag("caloRecHits","EcalRecHitsEE"))
towerMaker.hbheInput = 'caloRecHits'
towerMaker.hfInput = 'caloRecHits'
towerMaker.hoInput = 'caloRecHits'
paramMuons.MUONS.ProduceL1Muons = True
paramMuons.MUONS.ProduceL3Muons = True

