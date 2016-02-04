import FWCore.ParameterSet.Config as cms

process = cms.Process("EnergyScale")

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

## Select good jets
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import *
## use this selection of calorimeter based jets
process.goodJets = selectedPatJets.clone(
    src="cleanPatJets",
    cut = 'abs(eta) < 3 & pt > 30. &'
    'emEnergyFraction > 0.01       &'
    'jetID.fHPD < 0.98             &'
    'jetID.n90Hits > 1'    
)

## use this selection of particle flow jets
#process. goodJets   = selectedPatJets.clone(
#    src = 'cleanPatJetsAK5PF',
#    cut = 'abs(eta) < 2.4 & pt > 30.          &'
#    'chargedHadronEnergyFraction > 0.0  &'
#    'neutralHadronEnergyFraction/corrFactor("raw") < 0.99 &'
#    'chargedEmEnergyFraction/corrFactor("raw")     < 0.99 &'
#    'neutralEmEnergyFraction/corrFactor("raw")     < 0.99 &'
#    'chargedMultiplicity > 0            &'
#    'nConstituents > 0'
#)

## Analyze jets
from PhysicsTools.PatExamples.PatJetAnalyzer_cfi import analyzePatJets
process.Uncorrected = analyzePatJets.clone(src = 'goodJets', corrLevel='raw')
process.L2Relative  = analyzePatJets.clone(src = 'goodJets', corrLevel='rel')
process.L3Absolute  = analyzePatJets.clone(src = 'goodJets', corrLevel='abs')

## Define output file
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzeJetEnergyScale.root')
)

process.p = cms.Path(
    process.goodJets    * 
    process.Uncorrected *
    process.L2Relative  *
    process.L3Absolute

)
