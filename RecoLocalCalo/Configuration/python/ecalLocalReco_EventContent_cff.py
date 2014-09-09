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

