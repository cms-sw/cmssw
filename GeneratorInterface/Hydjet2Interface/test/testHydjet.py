import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("GeneratorInterface.Hydjet2Interface.hydjet2Default_cfi")

# Event output
#process.load("Configuration.EventContent.EventContent_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
	generator = cms.PSet(
		initialSeed = cms.untracked.uint32(123456789),
		engineName = cms.untracked.string('HepJamesRandom')
	)
)

process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(100)
)
'''
process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
	ignoreTotal=cms.untracked.int32(0),
	oncePerEventMode = cms.untracked.bool(False)
)
'''
process.ana = cms.EDAnalyzer('Hydjet2Analyzer')

process.TFileService = cms.Service('TFileService',
	fileName = cms.string('treefile.root')
)

process.p = cms.Path(process.generator*process.ana)


