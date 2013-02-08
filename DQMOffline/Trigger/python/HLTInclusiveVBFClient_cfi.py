import FWCore.ParameterSet.Config as cms

hltInclusiveVBFClient = cms.EDAnalyzer(
    "HLTInclusiveVBFClient",
    processname = cms.string("HLT"),
    DQMDirName  = cms.string("HLT/InclusiveVBF"),
    hltTag      = cms.string("HLT")
)

