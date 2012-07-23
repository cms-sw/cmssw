import FWCore.ParameterSet.Config as cms

#restarting from ECAL RAW to reconstruct amplitudes and energies
# create uncalib recHit collections
from Configuration.StandardSequences.RawToDigi_Data_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
ecalUncalibRecHitSequence = cms.Sequence(ecalGlobalUncalibRecHit * ecalDetIdToBeRecovered)
        
#process.ecalRecHit.laserCorrection=cms.bool(ApplyLaser)
# can add a flag for ICs?
#no switch in standard recHit producer to apply new intercalibrations
uncalibRecHitSeq = cms.Sequence( (ecalDigis + ecalPreshowerDigis) * ecalUncalibRecHitSequence)

sandboxSeq  = uncalibRecHitSeq
