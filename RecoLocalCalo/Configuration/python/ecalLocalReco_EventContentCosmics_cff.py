import FWCore.ParameterSet.Config as cms

# AOD content
ecalLocalRecoAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
# RECO content
# .. calibrated RecHits
ecalLocalRecoRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ecalPreshowerRecHit_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_ecalCompactTrigPrim_*_*',
        'keep ESDataFramesSorted_ecalPreshowerDigis_*_*',
        'keep EBSrFlagsSorted_ecalDigis__*',
        'keep EESrFlagsSorted_ecalDigis__*')
)
ecalLocalRecoRECO.outputCommands.extend(ecalLocalRecoAOD.outputCommands)

# Full Event content 
# .. EB + EE uncalibrated recHits
# .. calibrated RecHits
ecalLocalRecoFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalFixedAlphaBetaFitUncalibRecHit_*_*')
)
ecalLocalRecoFEVT.outputCommands.extend(ecalLocalRecoRECO.outputCommands)
