import FWCore.ParameterSet.Config as cms

RecoHiCentralityFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentrality*_hiCentrality_*_*',
                                           'keep recoClusterCompatibility*_hiClusterCompatibility_*_*')
    )

RecoHiCentralityRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentrality*_hiCentrality_*_*',
                                           'keep recoClusterCompatibility*_hiClusterCompatibility_*_*')
    )

RecoHiCentralityAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCentrality*_hiCentrality_*_*',
                                           'keep recoClusterCompatibility*_hiClusterCompatibility_*_*')
    )
