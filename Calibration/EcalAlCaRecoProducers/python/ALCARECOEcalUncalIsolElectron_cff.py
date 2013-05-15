import FWCore.ParameterSet.Config as cms

from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff import *
from Configuration.StandardSequences.RawToDigi_Data_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
#provides ecalRecHitSequence and ecalUncalibRecHitSequence

#ecalUncalibRecHitSequence = cms.Sequence(ecalGlobalUncalibRecHit * ecalDetIdToBeRecovered)
uncalibRecHitSeq          = cms.Sequence((ecalDigis+ecalPreshowerDigis) * ecalUncalibRecHitSequence)

seqALCARECOEcalUncalElectron = cms.Sequence( uncalibRecHitSeq )


############################################### FINAL SEQUENCES
seqALCARECOEcalUncalZElectron   = cms.Sequence( ZSkimSeq   *
                                                (seqALCARECOEcalUncalElectron + ALCARECOEcalCalElectronPreSeq)
                                                )
seqALCARECOEcalUncalZSCElectron = cms.Sequence( ZSCSkimSeq *
                                                (seqALCARECOEcalUncalElectron + ALCARECOEcalCalElectronPreSeq)
                                                )
seqALCARECOEcalUncalWElectron   = cms.Sequence( WSkimSeq   *
                                                (seqALCARECOEcalUncalElectron + ALCARECOEcalCalElectronPreSeq)
                                                ) 
