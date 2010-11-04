import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V13::All'

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.demo = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK5PF'),
	globalTag      = cms.untracked.string('START38_V13'),  
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True)
)

process.p = cms.Path(process.demo)
