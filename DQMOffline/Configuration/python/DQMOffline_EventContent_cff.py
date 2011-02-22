import FWCore.ParameterSet.Config as cms

MEtoEDMConverterFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*')
)
MEtoEDMConverterRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*')
)
MEtoEDMConverterAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

