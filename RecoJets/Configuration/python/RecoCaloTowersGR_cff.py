import FWCore.ParameterSet.Config as cms

#
# create GR calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *

from RecoJets.JetProducers.CaloTowerSchemeBWithHO_cfi import *

from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
towerMaker.HBThreshold = cms.double(0.6)

recoCaloTowersGR = cms.Sequence(towerMaker+towerMakerWithHO)

