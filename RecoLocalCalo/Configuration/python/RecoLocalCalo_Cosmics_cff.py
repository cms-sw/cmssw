import FWCore.ParameterSet.Config as cms

#
# Ecal part
#
from RecoLocalCalo.Configuration.ecalLocalRecoSequenceCosmics_cff import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

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
# sequence CaloLocalReco and CaloGlobalReco
#
calolocalreco = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence)
hbheprereco.firstSample = 1
hbheprereco.samplesToAdd = 8
hbheprereco.correctForTimeslew = False
hbheprereco.correctForPhaseContainment = False
horeco.firstSample = 1
horeco.samplesToAdd = 8
horeco.correctForTimeslew = False
horeco.correctForPhaseContainment = False
hfreco.firstSample = 1
hfreco.samplesToAdd = 4
hfreco.correctForTimeslew = False
hfreco.correctForPhaseContainment = False
#--- special temporary DB-usage unsetting 
hbheprereco.tsFromDB = False
hfreco.tsFromDB = False
horeco.tsFromDB = False
hfreco.digiTimeFromDB = False
hbheprereco.recoParamsFromDB = cms.bool(False)
horeco.recoParamsFromDB      = cms.bool(False)
hfreco.recoParamsFromDB      = cms.bool(False)
#zdcreco.firstSample = 1
#zdcreco.samplesToAdd = 8
zdcreco.correctForTimeslew = True
zdcreco.correctForPhaseContainment = True
zdcreco.correctionPhaseNS = 10.
#caloglobalreco = cms.Sequence(hcalGlobalRecoSequence)

#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoNZS = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence+hcalLocalRecoSequenceNZS) 
