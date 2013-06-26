import FWCore.ParameterSet.Config as cms

hltMonBTagMuClient = cms.EDAnalyzer('HLTMonBTagClient',
    monitorName             = cms.string('HLT/HLTMonBJet'),
    pathName                = cms.string('HLT_BTagMu_Jet10U'),
    storeROOT               = cms.untracked.bool(False),
    updateLuminosityBlock   = cms.untracked.bool(False),
    updateRun               = cms.untracked.bool(False),
    updateJob               = cms.untracked.bool(False),
    outputFile              = cms.untracked.string('HLTMonBTag.root')
)
