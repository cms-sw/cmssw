import FWCore.ParameterSet.Config as cms

# File: caloTowers.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for caloTowers.
# Assumes caloTowers are in event. Puts caloTowersOpt in the event, which
# assumes RecHits.
from DQMOffline.JetMET.caloTowers_cfi import *
from RecoMET.Configuration.CaloTowersOptForMET_cff import *
analyzecaloTowers = cms.Sequence(caloTowersMETOptRec*towerOptAnalyzer*towerSchemeBAnalyzer)
analyzecaloTowersDQM = cms.Sequence(towerSchemeBAnalyzer)

