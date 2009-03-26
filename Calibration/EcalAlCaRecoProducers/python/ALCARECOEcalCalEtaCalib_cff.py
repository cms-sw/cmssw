import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco eta calibration:
#------------------------------------------------
from Calibration.EcalAlCaRecoProducers.alcastreamEcalEtaCalib_cff import *
import RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi

#apply laser corrections

ecalEtaCorrected =  RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi.ecalRecHit.clone(
            doEnergyScale = cms.bool(False),
            doIntercalib = cms.bool(False),
            EERecHitCollection = cms.InputTag("hltAlCaEtaRegRecHits","etaEcalRecHitsEE"),
            EBRecHitCollection = cms.InputTag("hltAlCaEtaRegRecHits","etaEcalRecHitsEB"),
            doLaserCorrections = cms.bool(True),
            EBRecalibRecHitCollection = cms.string('etaEcalRecHitsEB'),
            EERecalibRecHitCollection = cms.string('etaEcalRecHitsEE')
)


seqALCARECOEcalCalEtaCalib = cms.Sequence(ecaletaCalibHLT*ecalEtaCorrected)
