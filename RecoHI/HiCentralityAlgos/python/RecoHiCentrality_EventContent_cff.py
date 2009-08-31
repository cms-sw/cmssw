import FWCore.ParameterSet.Config as cms

RecoHiCentralityFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentralityCollection_hiCentrality_*_*')
    )

RecoHiCentralityRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentralityCollection_hiCentrality_*_*')
    )

RecoHiCentralityAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentralityCollection_hiCentrality_*_*')
    )
