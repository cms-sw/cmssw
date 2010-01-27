import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Configuration.ecalLocalReco_EventContent_cff import *
#
# start with HCAL part
#
#FEVT
RecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*', 
        'keep *_hfreco_*_*', 
        'keep *_horeco_*_*',
        'keep HBHERecHitsSorted_hbherecoMB_*_*',
        'keep HORecHitsSorted_horecoMB_*_*',
        'keep HFRecHitsSorted_hfrecoMB_*_*',
        'keep ZDCDataFramesSorted_*Digis_*_*',
        'keep ZDCRecHitsSorted_*_*_*',
                                           'keep *_castorreco_*_*',
                                           'keep *_CastorTowerReco_*_*',
                                           'keep *_CastorClusterRecoAntiKt07_*_*',
                                           'keep *_CastorJetEgammaRecoAntiKt07_*_*'                                           
        )
)
#RECO content
RecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*', 
        'keep *_hfreco_*_*', 
        'keep *_horeco_*_*',
        'keep HBHERecHitsSorted_hbherecoMB_*_*',
        'keep HORecHitsSorted_horecoMB_*_*',
        'keep HFRecHitsSorted_hfrecoMB_*_*',
        'keep ZDCDataFramesSorted_*Digis_*_*',
        'keep ZDCRecHitsSorted_*_*_*',
                                           'keep *_castorreco_*_*',
                                           'keep *_CastorTowerReco_*_*',
                                           'keep *_CastorClusterRecoAntiKt07_*_*',
                                           'keep *_CastorJetEgammaRecoAntiKt07_*_*'                                           
        )
)
#AOD content
RecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep *_castorreco_*_*',
    'keep *_CastorTowerReco_*_*',
    'keep *_CastorClusterRecoAntiKt07_*_*',
    'keep *_CastorJetEgammaRecoAntiKt07_*_*'                                           
    )
)
RecoLocalCaloFEVT.outputCommands.extend(ecalLocalRecoFEVT.outputCommands)
RecoLocalCaloRECO.outputCommands.extend(ecalLocalRecoRECO.outputCommands)
RecoLocalCaloAOD.outputCommands.extend(ecalLocalRecoAOD.outputCommands)

