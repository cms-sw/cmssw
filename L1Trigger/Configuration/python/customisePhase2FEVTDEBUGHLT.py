import FWCore.ParameterSet.Config as cms

def customisePhase2FEVTDEBUGHLT(process):
    process.source.inputCommands = cms.untracked.vstring("keep *","drop l1tPFCandidates_*_*_RECO")
    return process
