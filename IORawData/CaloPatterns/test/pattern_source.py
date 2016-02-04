import FWCore.ParameterSet.Config as cms

process = cms.Process("PatternSource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'DESIGN_3X_V14::All'
process.source = cms.Source("EmptySource")

process.load("IORawData/CaloPatterns/HcalPatternSource_cfi")

process.p = cms.Path(process.hcalPatternSource)


