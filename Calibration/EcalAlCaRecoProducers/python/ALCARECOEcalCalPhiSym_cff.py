import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for phi symmetry calibration:
#------------------------------------------------
# create sequence for rechit filtering for phi symmetry calibration
from Calibration.EcalAlCaRecoProducers.alcastreamEcalPhiSym_cff import *
import RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi

#apply laser corrections

ecalPhiSymCorrected =  RecoLocalCalo.EcalRecProducers.ecalRecalibRecHit_cfi.ecalRecHit.clone(
            doEnergyScale = cms.bool(False),
            doIntercalib = cms.bool(False),
            EERecHitCollection = cms.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEE"),
            EBRecHitCollection = cms.InputTag("hltAlCaPhiSymStream","phiSymEcalRecHitsEB"),
            doLaserCorrections = cms.bool(True),
            EBRecalibRecHitCollection = cms.string('phiSymEcalRecHitsEB'),
            EERecalibRecHitCollection = cms.string('phiSymEcalRecHitsEE')
)

seqALCARECOEcalCalPhiSym = cms.Sequence(ecalphiSymHLT*ecalPhiSymCorrected)

