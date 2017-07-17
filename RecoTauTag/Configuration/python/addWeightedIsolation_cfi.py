import FWCore.ParameterSet.Config as cms

def addWeightedIsolation(process):
    process.load("RecoTauTag.Configuration.HPSPFTaus_cff")
    process.hpsPFTauMVAIsolation2Seq+=process.hpsPFTauMVA3IsolationNeutralIsoPtSumWeight

    return process
