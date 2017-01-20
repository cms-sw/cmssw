import FWCore.ParameterSet.Config as cms

from DQMOffline.Hcal.CaloTowersParam_cfi import *
import DQMOffline.Hcal.CaloTowersParam_cfi

from DQMOffline.Hcal.HcalRecHitParam_cfi import *
import DQMOffline.Hcal.HcalRecHitParam_cfi

from DQMOffline.Hcal.HcalNoiseRatesParam_cfi import *
import DQMOffline.Hcal.HcalNoiseRatesParam_cfi

AllCaloTowersDQMOffline = DQMOffline.Hcal.CaloTowersParam_cfi.calotowersAnalyzer.clone()
RecHitsDQMOffline       = DQMOffline.Hcal.HcalRecHitParam_cfi.hcalRecHitsAnalyzer.clone()
NoiseRatesDQMOffline    = DQMOffline.Hcal.HcalNoiseRatesParam_cfi.hcalNoiseRates.clone()

HcalDQMOfflineSequence = cms.Sequence(NoiseRatesDQMOffline*RecHitsDQMOffline*AllCaloTowersDQMOffline)
#HcalDQMOfflineSequence = cms.Sequence(NoiseRatesDQMOffline*AllCaloTowersDQMOffline)

_phase2_HcalDQMOfflineSequence = HcalDQMOfflineSequence.copyAndExclude([NoiseRatesDQMOffline])
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith(HcalDQMOfflineSequence, _phase2_HcalDQMOfflineSequence)
