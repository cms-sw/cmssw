import FWCore.ParameterSet.Config as cms

# Full Event content 
# .. EB + EE uncalibrated recHits
# .. calibrated RecHits
ecalLocalRecoFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalFixedAlphaBetaFitUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*',
        'keep ESDataFramesSorted_ecalPreshowerDigis_*_*'
        )
)
# RECO content
# .. calibrated RecHits
ecalLocalRecoRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*',
        'keep ESDataFramesSorted_ecalPreshowerDigis_*_*',
        'keep *_ecalCompactTrigPrim_*_*',
        'keep *_ecalTPSkim_*_*'
        )
)
# AOD content
# .. nothing
ecalLocalRecoAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

