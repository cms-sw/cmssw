import FWCore.ParameterSet.Config as cms

#
# create calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeBWithHO_cfi import *
caloTowersRecTask = cms.Task(towerMaker , towerMakerWithHO)
caloTowersRec = cms.Sequence(caloTowersRecTask)

