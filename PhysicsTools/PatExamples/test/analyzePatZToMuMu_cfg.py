import FWCore.ParameterSet.Config as cms

## Declare process
process = cms.Process("MuonAna")

## Declare input
from PhysicsTools.PatExamples.samplesCERN_cff import *

process.source = cms.Source("PoolSource",
  fileNames = zjetsTracks
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

## Message logger configuration
process.MessageLogger = cms.Service("MessageLogger")

## Selection of good muons
from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
process.goodMuons = selectedPatMuons.clone(
    src="cleanPatMuons",
    cut='pt>20. & abs(eta)<2.1 & (trackIso+caloIso)/pt<0.1',
)

## Monitor muons
process.load("PhysicsTools.PatExamples.PatZToMuMuAnalyzer_cfi")
process.analyzeZToMuMu.muons = 'goodMuons'

## Define output file
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzePatZToMuMu.root')
)

process.p = cms.Path(
    process.goodMuons *
    process.analyzeZToMuMu
)

