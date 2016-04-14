import FWCore.ParameterSet.Config as cms

dumpTauVariables = cms.EDProducer(
    "CandViewNtpProducer",
    ## can be selectedTausCandidates or slimmedTaus from miniAOD
    src = cms.InputTag("slimmedTaus"),
    ## prefix for variables for the dump
    prefix     = cms.untracked.string(""),
    ## allow access to function calls from derived classes
    lazyParser = cms.untracked.bool(True),
    ## add run, event number and lumi section
    eventInfo  = cms.untracked.bool(True),
    ## define variables to be dumped
    variables  = cms.VPSet()
    )
