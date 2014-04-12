import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:PATLayer1_Output.fromAOD_full.root'
  )
)

process.MessageLogger = cms.Service("MessageLogger")

process.analyzeZjetsJets = cms.EDAnalyzer("PatZjetsJetAnalyzer",
  src = cms.untracked.InputTag("cleanLayer1Jets")
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('analyzeZjetsJets.root')
                                   )

process.p = cms.Path(process.analyzeZjetsJets)

