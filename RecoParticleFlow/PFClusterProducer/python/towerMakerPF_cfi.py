import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.CaloTowerSchemeB_cfi
towerMakerPF = RecoJets.JetProducers.CaloTowerSchemeB_cfi.towerMaker.clone()
# replace towerMakerPF.HBThreshold = 0.
towerMakerPF.HOThreshold = 999999
towerMakerPF.HESThreshold = 0.
towerMakerPF.HEDThreshold = 0.
#replace towerMakerPF.HF1Threshold = 0
#replace towerMakerPF.HF2Threshold = 0
towerMakerPF.EBThreshold = 999999
towerMakerPF.EEThreshold = 999999
#replace towerMakerPF.EBSumThreshold =-1000 // GeV, -1000 means cut not used 
#replace towerMakerPF.EESumThreshold =-1000 // GeV, -1000 means cut not used 
#replace towerMakerPF.HcalThreshold = -1000 // GeV, -1000 means cut not used 
# replace towerMakerPF.EcutTower = -1000     // GeV, -1000 means cut not used
towerMakerPF.UseHO = False
towerMakerPF.AllowMissingInputs = True

