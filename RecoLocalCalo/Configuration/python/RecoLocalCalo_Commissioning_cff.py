import FWCore.ParameterSet.Config as cms

#
# Ecal part
#
from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *
#defines a sequence ecalLocalRecoSequence
#
# Hcal part
#
# calo geometry
#
# changed by tommaso. now the calibrations are read from Configuration/StaqndardSequences/data/*Conditions.cff
#
# HCAL calibrations
#include "CalibCalorimetry/HcalPlugins/data/hardwired_conditions.cfi"
#HCAL reconstruction
from RecoLocalCalo.Configuration.hcalLocalReco_cff import *
from RecoLocalCalo.Configuration.hcalGlobalReco_cff import *
#
hfreco.firstSample = 2
hfreco.samplesToAdd = 2
#--- special temporary DB-usage unsetting
hfreco.tsFromDB = False
hfreco.digiTimeFromDB = False
#
# sequence CaloLocalReco and CaloGlobalReco
#
calolocalrecoTask = cms.Task(ecalLocalRecoTask,hcalLocalRecoTask)
calolocalreco = cms.Sequence(calolocalrecoTask)

#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoTaskNZS = cms.Task(ecalLocalRecoTask,hcalLocalRecoTask,hcalLocalRecoTaskNZS) 
calolocalrecoNZS = cms.Sequence(calolocalrecoTaskNZS) 
