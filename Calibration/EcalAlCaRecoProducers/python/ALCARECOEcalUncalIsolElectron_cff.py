import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOEcalUncalZElectronHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
    eventSetupPathsKey = 'EcalUncalZElectron',
    throw = False # tolerate triggers stated above, but not available
)
ALCARECOEcalUncalZSCElectronHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
    eventSetupPathsKey = 'EcalUncalZSCElectron', 
   throw = False # tolerate triggers stated above, but not available
)
ALCARECOEcalUncalWElectronHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
    eventSetupPathsKey = 'EcalUncalWElectron',
    throw = False # tolerate triggers stated above, but not available
)

#restarting from ECAL RAW to reconstruct amplitudes and energies
# create uncalib recHit collections
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff import *
from Configuration.StandardSequences.RawToDigi_Data_cff import ecalDigis,ecalPreshowerDigis
from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *

from RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi import *
ecalUncalibRecHitSequence53X = cms.Sequence(ecalGlobalUncalibRecHit * ecalDetIdToBeRecovered)

ecalAndPreshowerDigisForUncalibRecHitSeqTask = cms.Task(ecalDigis, ecalPreshowerDigis)
uncalibRecHitSeq = cms.Sequence(ecalUncalibRecHitSequence, ecalAndPreshowerDigisForUncalibRecHitSeqTask)

ALCARECOEcalUncalElectronECALSeq = cms.Sequence( uncalibRecHitSeq )

############################################### FINAL SEQUENCES
# sequences used in AlCaRecoStreams_cff.py
seqALCARECOEcalUncalZElectron   = cms.Sequence(ALCARECOEcalUncalZElectronHLT * ZeeSkimFilterSeq  * ALCARECOEcalUncalElectronECALSeq * ALCARECOEcalCalElectronNonECALSeq)
seqALCARECOEcalUncalZSCElectron = cms.Sequence(ALCARECOEcalUncalZSCElectronHLT * ZSCSkimFilterSeq  * ALCARECOEcalUncalElectronECALSeq * ALCARECOEcalCalElectronNonECALSeq)
seqALCARECOEcalUncalWElectron   = cms.Sequence(ALCARECOEcalUncalWElectronHLT * WenuSkimFilterSeq * ALCARECOEcalUncalElectronECALSeq* ALCARECOEcalCalElectronNonECALSeq)
