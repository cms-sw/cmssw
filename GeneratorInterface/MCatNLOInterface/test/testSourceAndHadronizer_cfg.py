import FWCore.ParameterSet.Config as cms

process = cms.Process('Test')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("MCatNLOSource",
                            fileNames = cms.untracked.vstring('file:Z.events'),
                            processCode = cms.int32(-11361),
                            skipEvents=cms.untracked.uint32(0)

)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.generator = cms.EDFilter("Herwig6HadronizerFilter",
	comEnergy = cms.double(10000.0),
	useJimmy = cms.bool(False),
	doMPInteraction = cms.bool(False),

	herwigHepMCVerbosity = cms.untracked.bool(False),
	herwigVerbosity = cms.untracked.int32(1),
	printCards = cms.untracked.bool(True),
	maxEventsToPrint = cms.untracked.int32(0),

	crossSection = cms.untracked.double(-1.0),
	filterEfficiency = cms.untracked.double(1.0),

	emulatePythiaStatusCodes = cms.untracked.bool(False),

        numTrialsMPI = cms.untracked.int32(1),

	HerwigParameters = cms.PSet(
		parameterSets = cms.vstring(
			'herwigMcatnlo'
		),
                herwigMcatnlo = cms.vstring(
			'PTMIN      = 0.5       ! minimum pt in hadronic jet'
		)
	)
)


process.RandomNumberGeneratorService.generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'


process.ProductionFilterSequence = cms.Sequence(process.generator)

process.generation_step = cms.Path(process.ProductionFilterSequence)

process.output = cms.OutputModule("PoolOutputModule",
                                  fileName = cms.untracked.string('mcatnloZee.root'),
                                  SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('generation_step')
    )
)

process.output_step = cms.EndPath(process.output)
