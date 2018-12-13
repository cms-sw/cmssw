import FWCore.ParameterSet.Config as cms

process = cms.Process('Test')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Herwig6GeneratorFilter",
	comEnergy = cms.double(7000.0),
	useJimmy = cms.bool(True),
	doMPInteraction = cms.bool(False),

	herwigHepMCVerbosity = cms.untracked.bool(False),
	herwigVerbosity = cms.untracked.int32(0),
	printCards = cms.untracked.bool(True),
	maxEventsToPrint = cms.untracked.int32(1),

	crossSection = cms.untracked.double(-1.0),
	filterEfficiency = cms.untracked.double(1.0),

	emulatePythiaStatusCodes = cms.untracked.bool(True),

        ParticleSpectrumFileName = cms.untracked.string('GeneratorInterface/Herwig6Interface/test/softsusy317_isajet764_LM5_GUTlp211eq0.01.hw65'),              
        readParticleSpecFile     = cms.untracked.bool(True),

        HerwigParameters = cms.PSet(
		parameterSets = cms.vstring(
                        'jimmyCMSdefault7TeV',
                        'softsusyIsajetRPV'
		),
                jimmyCMSdefault7TeV = cms.vstring(
                        'MODPDF(1)  = 10041      ! PDF set according to LHAGLUE', 
                        'MODPDF(2)  = 10041      ! CTEQ6L', 
                        'JMUEO      = 1          ! multiparton interaction model', 
                        'PTJIM      = 4.040      ! 2.8x(sqrt(s)/1.8TeV)^0.27 @ 7 TeV', 
                        'JMRAD(73)  = 1.8        ! inverse proton radius squared', 
                        'PRSOF      = 0.0        ! prob. of a soft underlying event'
                ),
                softsusyIsajetRPV = cms.vstring(
                         'IPROC = 4000            !resonant RPV slepton production (4010-50)',
                )
	)
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

process.generation_step = cms.Path(process.ProductionFilterSequence)

process.output = cms.OutputModule("PoolOutputModule",
	#fileName = cms.untracked.string('herwigHZZ4mu.root'),
	fileName = cms.untracked.string('softsusyIsajetRPV.root'),
	SelectEvents = cms.untracked.PSet(
		SelectEvents = cms.vstring('generation_step')
	)
)

process.output_step = cms.EndPath(process.output)
