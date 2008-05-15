import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersRec.cff,v 1.3 2008/04/30 22:05:16 fedor Exp $
#
# create calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
caloTowersRec = cms.Sequence(towerMaker)

