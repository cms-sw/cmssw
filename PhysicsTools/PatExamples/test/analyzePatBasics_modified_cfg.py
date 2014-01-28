import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:patTuple.root'
  )
)

process.MessageLogger = cms.Service("MessageLogger")

## using the process.load function
#process.load("PhysicsTools/PatExamples/PatBasicAnalyzer_cfi")
## using import and making the module known to the process afterwards
from PhysicsTools.PatExamples.PatBasicAnalyzer_cfi import analyzeBasicPat
process.analyzeBasicPat = analyzeBasicPat

## cloning an existing module
process.analyzeBasicPat2= analyzeBasicPat.clone()

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzePatBasics.root')
)

process.p = cms.Path(
    process.analyzeBasicPat *
    process.analyzeBasicPat2
)

