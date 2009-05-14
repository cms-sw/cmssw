import FWCore.ParameterSet.Config as cms

def customise(process):

    process.options.wantSummary = cms.untracked.bool(True)

    return(process)
