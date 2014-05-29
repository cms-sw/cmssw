
import FWCore.ParameterSet.Config as cms

#Full Event content 
hgcalLocalRecoFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_HGCalUncalibRecHit_*_*', 
        'keep *_HGCRecHit_*_*')
)
#RECO content
hgcalLocalRecoRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_HGCRecHit_*_*',
        )
)
#AOD content
hgcalLocalRecoAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        )
)

