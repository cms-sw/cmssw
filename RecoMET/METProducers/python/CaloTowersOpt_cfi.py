import FWCore.ParameterSet.Config as cms

# File: CaloTowerOpt.cfi
# Author: B. Scurlock
# Date: 03.06.2008
#
# Form uncorrected Missing ET from Calorimeter Towers and store into event as a CaloMET
# product
# Creates new calotowers with optimized Energy thresholds for MET.
# === Modification: 09/30/08 by R. Remington
# === Made modifications to accomodate changes to towerMaker (done by A. Oehler)
# === Modification : 05/14/09 by R.Remington
# === Now cloning the central calotowermaker module and changing parameters for optimized thresholds instead of writing another independent module   

from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import *

calotoweroptmaker = calotowermaker.clone() 
calotoweroptmaker.UseHO = False
calotoweroptmaker.HBThreshold = cms.double(0.5)
calotoweroptmaker.HESThreshold = cms.double(0.7)
calotoweroptmaker.HEDThreshold = cms.double(0.5)

calotoweroptmakerWithHO = calotoweroptmaker.clone()
calotoweroptmakerWithHO.UseHO = True


