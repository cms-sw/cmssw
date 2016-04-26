import FWCore.ParameterSet.Config as cms

#
# Ecal part
#
from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

#defines a sequence ecalLocalRecoSequence
#
# Hcal part
#
# calo geometry
#
# changed by tommaso. now the calibrations are read from Configuration/StaqndardSequences/data/*Conditions.cff
#
#HCAL reconstruction
from RecoLocalCalo.Configuration.hcalLocalReco_cff import *
from RecoLocalCalo.Configuration.hcalGlobalReco_cff import *
#
# sequence CaloLocalReco and CaloGlobalReco
#
calolocalreco = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence)
caloglobalreco = cms.Sequence(hcalGlobalRecoSequence)

from RecoLocalCalo.HcalRecProducers.HcalHitSelection_cfi import *
reducedHcalRecHitsSequence = cms.Sequence( reducedHcalRecHits )

#
# R.Ofierzynski (29.Oct.2009): add NZS sequence
#
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *
calolocalrecoNZS = cms.Sequence(ecalLocalRecoSequence+hcalLocalRecoSequence+hcalLocalRecoSequenceNZS) 

def _modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonHBHE( obj ):
    obj.digiLabel = cms.InputTag('simHcalDigis','HBHEUpgradeDigiCollection')

def _modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonHO( obj ):
    obj.digiLabel = cms.InputTag('simHcalDigis')

def _modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonHF( obj ):
    obj.digiLabel = cms.InputTag('simHcalDigis','HFUpgradeDigiCollection')

def _modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonZDC( obj ):
    obj.digiLabel = cms.InputTag('simHcalUnsuppressedDigis')
    obj.digiLabelhcal = cms.InputTag('simHcalUnsuppressedDigis')

def _modifyRecoLocalCaloConfigurationProcessForPhase2Common( theProcess ):
    theProcess.load("RecoLocalCalo.HcalRecProducers.HBHEUpgradeReconstructor_cfi")
    theProcess.load("RecoLocalCalo.HcalRecProducers.HFUpgradeReconstructor_cfi")
    theProcess.hcalLocalRecoSequence.replace(theProcess.hfreco,theProcess.hfUpgradeReco)
    theProcess.hcalLocalRecoSequence.remove(theProcess.hbhereco)
    theProcess.hcalLocalRecoSequence.replace(theProcess.hbheprereco,theProcess.hbheUpgradeReco)

def _modifyRecoLocalCaloConfigurationProcessForHGCal( theProcess ):
    theProcess.load("RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi")
    theProcess.load("RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi")
    theProcess.calolocalreco += theProcess.HGCalUncalibRecHit
    theProcess.calolocalreco += theProcess.HGCalRecHit

from Configuration.StandardSequences.Eras import eras
eras.phase2_common.toModify( hbhereco,  func=_modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonHBHE )
eras.phase2_common.toModify( horeco,  func=_modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonHO )
eras.phase2_common.toModify( hfreco,  func=_modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonHF )
eras.phase2_common.toModify( zdcreco, func=_modifyRecoLocalCaloConfigurationReconstructorsForPhase2CommonZDC )

modifyRecoLocalCaloConfigurationProcessForPhase2Common_ = eras.phase2_common.makeProcessModifier( _modifyRecoLocalCaloConfigurationProcessForPhase2Common )
modifyRecoLocalCaloConfigurationProcessForHGCal_ = eras.phase2_hgcal.makeProcessModifier( _modifyRecoLocalCaloConfigurationProcessForHGCal )

