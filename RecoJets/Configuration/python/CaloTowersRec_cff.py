import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersRec_cff.py,v 1.4 2008/05/19 20:56:52 rpw Exp $
#
# create calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeBWithHO_cfi import *
caloTowersRec = cms.Sequence(towerMaker + towerMakerWithHO)

