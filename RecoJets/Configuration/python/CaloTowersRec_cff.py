import FWCore.ParameterSet.Config as cms

# $Id: CaloTowersRec.cff,v 1.2 2008/03/06 16:10:29 fedor Exp $
#
# create calotowers here
#
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoJets.JetProducers.CaloTowerSchemeB_cfi import *
caloTowers = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMaker"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

caloTowersRec = cms.Sequence(towerMaker*caloTowers)

