import FWCore.ParameterSet.Config as cms

process = cms.Process("Jets")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    'file:patTuple.root'
  )
)

process.MessageLogger = cms.Service("MessageLogger")

## prepare jet collections
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets
process.goodCaloJets = selectedPatJets.clone(src='cleanPatJets', cut='pt>30 & abs(eta)<3')

## monitor jet collections
from PhysicsTools.PatExamples.PatJetAnalyzer_cfi import analyzePatJets
## modules for jet response
process.rawJets     = analyzePatJets.clone(corrLevel='Uncorrected')
process.relJets     = analyzePatJets.clone(corrLevel='L2Relative')
process.absJets     = analyzePatJets.clone(corrLevel='L3Absolute')
## modules to compare calo and pflow jets
process.caloJets    = analyzePatJets.clone(src='goodCaloJets')
process.pflowJets   = analyzePatJets.clone(src='goodCaloJets')
## modules for shift in JES
process.shiftedJets = analyzePatJets.clone(src='goodCaloJets')


process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzePatJets.root')
)

## process path
process.p = cms.Path(
    process.goodCaloJets * 
    process.rawJets   *
    process.relJets   *
    process.absJets   *
    process.caloJets  *
    process.pflowJets *
    process.shiftedJets
)
