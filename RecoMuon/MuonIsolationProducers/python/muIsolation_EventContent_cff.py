# The following comments couldn't be translated into the new config version:

#MuIsoDeposits

#MuIsoDeposits

#MuIsoDeposits

#old version (reduced set)

#standard set

import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
#define FEVT/REC/AOD pieces

#AOD part of the event
#cleaned-up, includes only the objects produced in the standard reco "muIsolation" sequence
RecoMuonIsolationAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

#RECO part of the event
RecoMuonIsolationRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_muIsoDepositTkDisplaced_*_*',
        'keep *_muIsoDepositCalByAssociatorTowersDisplaced_*_*', 
        'keep *_muIsoDepositCalByAssociatorHitsDisplaced_*_*', 
        'keep *_muIsoDepositJetsDisplaced_*_*', 
        'keep *_muGlobalIsoDepositCtfTk_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muGlobalIsoDepositJets_*_*')
)
RecoMuonIsolationRECO.outputCommands.extend(RecoMuonIsolationAOD.outputCommands)

#Full event
RecoMuonIsolationFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoMuonIsolationFEVT.outputCommands.extend(RecoMuonIsolationRECO.outputCommands)

#Full event
RecoMuonIsolationParamGlobal = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muParamGlobalIsoDepositGsTk_*_*', 
        'keep *_muParamGlobalIsoDepositCalEcal_*_*', 
        'keep *_muParamGlobalIsoDepositCalHcal_*_*', 
        'keep *_muParamGlobalIsoDepositCtfTk_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muParamGlobalIsoDepositJets_*_*')
)
