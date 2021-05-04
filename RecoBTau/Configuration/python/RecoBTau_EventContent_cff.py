import FWCore.ParameterSet.Config as cms

#AOD content
RecoBTauAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

#RECO content
RecoBTauRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoBTauRECO.outputCommands.extend(RecoBTauAOD.outputCommands)

#Full Event content 
RecoBTauFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoBTauFEVT.outputCommands.extend(RecoBTauRECO.outputCommands)
