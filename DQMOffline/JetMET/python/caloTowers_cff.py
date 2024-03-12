import FWCore.ParameterSet.Config as cms

# File: caloTowers.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for caloTowers.
# Assumes caloTowers are in event.
from DQMOffline.JetMET.caloTowers_cfi import *
analyzecaloTowersDQM = cms.Sequence(towerSchemeBAnalyzer)

# foo bar baz
# q7zlR8n0MY6uA
# z1nwgcDwqU4Cq
