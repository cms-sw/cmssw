import FWCore.ParameterSet.Config as cms

RecoHiCentralityFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentralitys_hiCentrality_*_*')
    )

RecoHiCentralityRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentralitys_hiCentrality_*_*')
    )

RecoHiCentralityAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentralitys_hiCentrality_*_*')
    )
