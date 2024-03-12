import FWCore.ParameterSet.Config as cms

hltMonBTagIPClient = cms.EDAnalyzer('HLTMonBTagClient',
    monitorName             = cms.string('HLT/HLTMonBJet'),
    pathName                = cms.string('HLT_BTagIP_Jet50U'),
    storeROOT               = cms.untracked.bool(False),
    updateLuminosityBlock   = cms.untracked.bool(False),
    updateRun               = cms.untracked.bool(False),
    updateJob               = cms.untracked.bool(False),
    outputFile              = cms.untracked.string('HLTMonBTag.root')
)
# foo bar baz
# WqhgPkKWbIvSm
