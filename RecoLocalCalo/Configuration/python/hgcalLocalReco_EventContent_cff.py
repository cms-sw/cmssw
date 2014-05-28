
import FWCore.ParameterSet.Config as cms

#Full Event content 
hgcalLocalRecoFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_*_HGCalUncalibRecHit_*',
        'keep *_*_HGCHEBRecHits_*',
        'keep *_*_HGCHEFRecHits_*',        
        'keep *_*_HGCEERecHits_*')
)
#RECO content
hgcalLocalRecoRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_*_HGCHEBRecHits_*',
        'keep *_*_HGCHEFRecHits_*',
        'keep *_*_HGCEERecHits_*'        
        )
)
#AOD content
hgcalLocalRecoAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        )
)

