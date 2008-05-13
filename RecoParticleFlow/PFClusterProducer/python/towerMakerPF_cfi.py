# The following comments couldn't be translated into the new config version:

# GeV, -1000 means cut not used 
# GeV, -1000 means cut not used 
import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.CaloTowerSchemeB_cfi
towerMakerPF = RecoJets.JetProducers.CaloTowerSchemeB_cfi.towerMaker.clone()
towerMakerPF.HBThreshold = 0.
towerMakerPF.HOThreshold = 999999
towerMakerPF.HESThreshold = 0.
towerMakerPF.HEDThreshold = 0.
towerMakerPF.HF1Threshold = 999999
towerMakerPF.HF2Threshold = 999999
towerMakerPF.EBThreshold = 999999
towerMakerPF.EEThreshold = 999999
towerMakerPF.EBSumThreshold = -1000
towerMakerPF.EESumThreshold = -1000
towerMakerPF.HcalThreshold = -1000 ## GeV, -1000 means cut not used 

towerMakerPF.EcutTower = -1000 ## GeV, -1000 means cut not used

towerMakerPF.UseHO = False
towerMakerPF.AllowMissingInputs = True

