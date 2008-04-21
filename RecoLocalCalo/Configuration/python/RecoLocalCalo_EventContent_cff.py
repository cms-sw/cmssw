import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.Configuration.ecalLocalReco_EventContent_cff import *
#
# start with HCAL part
#
#FEVT
RecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*', 
        'keep *_hfreco_*_*', 
        'keep *_horeco_*_*')
)
#RECO content
RecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*', 
        'keep *_hfreco_*_*', 
        'keep *_horeco_*_*')
)
#AOD content
RecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoLocalCaloFEVT.outputCommands.extend(ecalLocalRecoFEVT.outputCommands)
RecoLocalCaloRECO.outputCommands.extend(ecalLocalRecoRECO.outputCommands)
RecoLocalCaloAOD.outputCommands.extend(ecalLocalRecoAOD.outputCommands)

