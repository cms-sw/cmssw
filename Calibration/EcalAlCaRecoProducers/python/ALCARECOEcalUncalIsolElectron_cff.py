import FWCore.ParameterSet.Config as cms

#restarting from ECAL RAW to reconstruct amplitudes and energies
# create uncalib recHit collections
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff import *
from Configuration.StandardSequences.RawToDigi_Data_cff import *
from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *

from RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi import *
ecalUncalibRecHitSequence53X = cms.Sequence(ecalGlobalUncalibRecHit * ecalDetIdToBeRecovered)

uncalibRecHitSeq = cms.Sequence( (ecalDigis + ecalPreshowerDigis) * ecalUncalibRecHitSequence)

ALCARECOEcalUncalElectronECALSeq = cms.Sequence( uncalibRecHitSeq )

############################################### FINAL SEQUENCES
# sequences used in AlCaRecoStreams_cff.py
seqALCARECOEcalUncalZElectron   = cms.Sequence(ZeeSkimFilterSeq  * ALCARECOEcalCalElectronNonECALSeq * ALCARECOEcalUncalElectronECALSeq)
seqALCARECOEcalUncalZSCElectron = cms.Sequence(ZSCSkimFilterSeq  * ALCARECOEcalCalElectronNonECALSeq * ALCARECOEcalUncalElectronECALSeq)
seqALCARECOEcalUncalWElectron   = cms.Sequence(WenuSkimFilterSeq * ALCARECOEcalCalElectronNonECALSeq * ALCARECOEcalUncalElectronECALSeq)
