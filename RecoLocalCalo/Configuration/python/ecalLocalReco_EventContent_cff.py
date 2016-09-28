# The following comments couldn't be translated into the new config version:

#EB + EE Uncalibrated RecHits

#Calibrated RecHits

#Calibrated RecHits

import FWCore.ParameterSet.Config as cms

#Full Event content 
ecalLocalRecoFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ecalMultiFitUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*')
)
#RECO content
ecalLocalRecoRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*',
        'keep *_ecalCompactTrigPrim_*_*',
        'keep *_ecalTPSkim_*_*'
        )
)
#AOD content
ecalLocalRecoAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        )
)

#mods for timing
_phase2_timing_EcalOutputCommands = ['keep *_mix_EBTimeDigi_*',
                                     'keep *_mix_EETimeDigi_*', 
                                     'keep *_ecalDetailedTimeRecHit_*_*']

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( ecalLocalRecoFEVT, outputCommands = ecalLocalRecoFEVT.outputCommands + _phase2_timing_EcalOutputCommands )
phase2_timing.toModify( ecalLocalRecoRECO, outputCommands = ecalLocalRecoRECO.outputCommands + _phase2_timing_EcalOutputCommands )

