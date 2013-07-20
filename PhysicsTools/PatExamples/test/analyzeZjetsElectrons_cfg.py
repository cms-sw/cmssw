import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:PATLayer1_Output.fromAOD_full.root'
  )
)

process.MessageLogger = cms.Service("MessageLogger")

process.analyzeZjetsElectrons = cms.EDAnalyzer("PatZjetsElectronAnalyzer",
  src = cms.untracked.InputTag("cleanLayer1Electrons")
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('analyzeZjetsElectrons.root')
                                   )

process.p = cms.Path(process.analyzeZjetsElectrons)

