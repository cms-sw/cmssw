import FWCore.ParameterSet.Config as cms

from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff import *
from Configuration.StandardSequences.RawToDigi_Data_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *

ecalUncalibRecHitSequence = cms.Sequence(ecalGlobalUncalibRecHit * ecalDetIdToBeRecovered)
uncalibRecHitSeq          = cms.Sequence((ecalDigis+ecalPreshowerDigis) * ecalUncalibRecHitSequence)

seqALCARECOEcalUncalElectron = cms.Sequence( uncalibRecHitSeq )

seqALCARECOEcalUncalZElectron = cms.Sequence( tagGsfSeq * seqALCARECOEcalUncalElectron)
seqALCARECOEcalUncalWElectron = cms.Sequence( WSkimSeq  * seqALCARECOEcalUncalElectron) 
