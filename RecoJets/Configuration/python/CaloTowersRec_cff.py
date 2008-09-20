import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersRec_cff.py,v 1.4.2.1 2008/09/04 20:32:52 fedor Exp $
#
# create calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeBWithHO_cfi import *
caloTowersRec = cms.Sequence(towerMaker + towerMakerWithHO)

