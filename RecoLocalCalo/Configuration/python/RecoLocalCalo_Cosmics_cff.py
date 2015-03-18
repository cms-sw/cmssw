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
hbheprereco.puCorrMethod = 0 
hbheprereco.firstSample = 0
hbheprereco.samplesToAdd = 10
hbheprereco.correctForTimeslew = False
hbheprereco.correctForPhaseContainment = False
horeco.firstSample = 0
horeco.samplesToAdd = 10
horeco.correctForTimeslew = False
horeco.correctForPhaseContainment = False
hfreco.firstSample = 0
hfreco.samplesToAdd = 10 ### min(10,size) in the algo
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
