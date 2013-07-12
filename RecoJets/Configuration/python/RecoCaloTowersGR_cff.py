import FWCore.ParameterSet.Config as cms

# $Id: RecoCaloTowersGR_cff.py,v 1.5 2008/10/21 14:40:07 oehler Exp $
#
# create GR calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *

from RecoJets.JetProducers.CaloTowerSchemeBWithHO_cfi import *

from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
towerMaker.HBThreshold = cms.double(0.6)

recoCaloTowersGR = cms.Sequence(towerMaker+towerMakerWithHO)

