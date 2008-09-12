import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.CaloTowerSchemeB_cfi

towerMakerWithHO = RecoJets.JetProducers.CaloTowerSchemeB_cfi.towerMaker.clone()
towerMakerWithHO.UseHO = True



