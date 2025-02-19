import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco pi0 calibration:
#------------------------------------------------
from Calibration.EcalAlCaRecoProducers.alcastreamEcalPi0Calib_cff import *
import RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi

#apply laser corrections

ecalPi0Corrected =  RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi.ecalRecHit.clone(
            doEnergyScale = cms.bool(False),
            doIntercalib = cms.bool(False),
            EERecHitCollection = cms.InputTag("hltAlCaPi0RecHitsFilter","pi0EcalRecHitsEE"),
            EBRecHitCollection = cms.InputTag("hltAlCaPi0RecHitsFilter","pi0EcalRecHitsEB"),
            doLaserCorrections = cms.bool(True),
            EBRecalibRecHitCollection = cms.string('pi0EcalRecHitsEB'),
            EERecalibRecHitCollection = cms.string('pi0EcalRecHitsEE')
)


seqALCARECOEcalCalPi0Calib = cms.Sequence(ecalpi0CalibHLT*ecalPi0Corrected)
