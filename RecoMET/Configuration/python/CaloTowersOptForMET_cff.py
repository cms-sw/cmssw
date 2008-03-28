import FWCore.ParameterSet.Config as cms

# File: CaloMETOptcaloTowers.cff
# Author: B. Scurlock
# Date: 02.28.2008
#
# Form Missing ET from optimized calotowers. The HCALRecHit thresholds 
# are based on Feng Liu's optimization study.
# IMPORTANT: this configuration assumes that RecHits are in the event
# reconstruct CaloRecHits and create calotowers here
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloTowersOpt_cfi import *
caloTowersOpt = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("calotoweroptmaker"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

caloTowersMETOptRec = cms.Sequence(calotoweroptmaker*caloTowersOpt)

