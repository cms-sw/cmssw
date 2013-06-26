import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersRec_cff.py,v 1.5 2008/09/20 17:41:45 oehler Exp $
#
# create calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeBWithHO_cfi import *
caloTowersRec = cms.Sequence(towerMaker + towerMakerWithHO)

