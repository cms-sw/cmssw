import FWCore.ParameterSet.Config as cms

#
# Ecal part
#
from RecoLocalCalo.Configuration.ecalLocalRecoSequenceCosmics_cff import *
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
hbhereco.firstSample = 1
hbhereco.samplesToAdd = 8
hbhereco.correctForTimeslew = True
hbhereco.correctForPhaseContainment = True
hbhereco.correctionPhaseNS = 10.0
horeco.firstSample = 1
horeco.samplesToAdd = 8
horeco.correctForTimeslew = True
horeco.correctForPhaseContainment = True
horeco.correctionPhaseNS = 10.
hfreco.firstSample = 1
hfreco.samplesToAdd = 8
hfreco.correctForTimeslew = True
hfreco.correctForPhaseContainment = True
hfreco.correctionPhaseNS = 10.
zdcreco.firstSample = 1
zdcreco.samplesToAdd = 8
zdcreco.correctForTimeslew = True
zdcreco.correctForPhaseContainment = True
zdcreco.correctionPhaseNS = 10.

#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoNZS = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence+hcalLocalRecoSequenceNZS) 
