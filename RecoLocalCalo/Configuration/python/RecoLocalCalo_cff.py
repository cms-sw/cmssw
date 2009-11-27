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
#
# sequence CaloLocalReco
#
calolocalreco = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence)

#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoNZS = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence+hcalLocalRecoSequenceNZS) 
