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

def _modifyRecoLocalCaloConfigurationProcessForPhase2Common( theProcess ):
    from RecoLocalCalo.HcalRecProducers.HBHEUpgradeReconstructor_cfi import hbheUpgradeReco as _hbheUpgradeReco
    from RecoLocalCalo.HcalRecProducers.HFUpgradeReconstructor_cfi import hfUpgradeReco as _hfUpgradeReco
    theProcess.hbheUpgradeReco = _hbheUpgradeReco
    theProcess.hfUpgradeReco = _hfUpgradeReco
    theProcess.hcalLocalRecoSequence.replace(theProcess.hfreco,theProcess.hfUpgradeReco)
    theProcess.hcalLocalRecoSequence.remove(theProcess.hbhereco)
    theProcess.hcalLocalRecoSequence.replace(theProcess.hbheprereco,theProcess.hbheUpgradeReco)

def _modifyRecoLocalCaloConfigurationProcessForHGCal( theProcess ):
    from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import HGCalUncalibRecHit as _HGCalUncalibRecHit
    from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import HGCalRecHit as _HGCalRecHit
    theProcess.HGCalUncalibRecHit = _HGCalUncalibRecHit
    theProcess.HGCalRecHit = _HGCalRecHit
    theProcess.calolocalreco += theProcess.HGCalUncalibRecHit
    theProcess.calolocalreco += theProcess.HGCalRecHit

from Configuration.StandardSequences.Eras import eras

eras.phase2_common.toModify( hbheprereco, digiLabel = cms.InputTag('simHcalDigis','HBHEUpgradeDigiCollection') )
eras.phase2_common.toModify( horeco, digiLabel = cms.InputTag('simHcalDigis') )
eras.phase2_common.toModify( hfreco, digiLabel = cms.InputTag('simHcalDigis','HFUpgradeDigiCollection') )
eras.phase2_common.toModify( zdcreco, digiLabel = cms.InputTag('simHcalUnsuppressedDigis'), digiLabelhcal = cms.InputTag('simHcalUnsuppressedDigis') )
modifyRecoLocalCaloConfigurationProcessForPhase2Common_ = eras.phase2_common.makeProcessModifier( _modifyRecoLocalCaloConfigurationProcessForPhase2Common )
modifyRecoLocalCaloConfigurationProcessForHGCal_ = eras.phase2_hgcal.makeProcessModifier( _modifyRecoLocalCaloConfigurationProcessForHGCal )

