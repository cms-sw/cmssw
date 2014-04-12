import FWCore.ParameterSet.Config as cms

process = cms.Process("EnergyShift")

## Declare input
from PhysicsTools.PatExamples.samplesCERN_cff import *

process.source = cms.Source("PoolSource",
  fileNames = ttbarJets
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

## Message logger configuration
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## Configure jet energy scaling module
process.load("PhysicsTools.PatExamples.JetEnergyShift_cfi")
process.scaledJets.scaleFactor =  1.0

## Select good jets
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import *
process.goodJets = selectedPatJets.clone(
    src="scaledJets:cleanPatJets",
    cut = 'abs(eta) < 3 & pt > 30. &'
    'emEnergyFraction > 0.01       &'
    'jetID.fHPD < 0.98             &'
    'jetID.n90Hits > 1'    
    )

## Analyze jets
process.load("PhysicsTools.PatExamples.PatJetAnalyzer_cfi")
process.analyzePatJets.src = 'goodJets'

## Define output file
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzeEnergyShiftedJets.root')
)

process.p = cms.Path(
    process.scaledJets *
    process.goodJets   * 
    process.analyzePatJets
)
