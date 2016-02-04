import FWCore.ParameterSet.Config as cms

process = cms.Process('Test')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(500))

process.source = cms.Source("Herwig6Source",
	useJimmy = cms.untracked.bool(True),
	doMPInteraction = cms.untracked.bool(True),

	herwigHepMCVerbosity = cms.untracked.bool(False),
	herwigVerbosity = cms.untracked.int32(0),
	printCards = cms.untracked.bool(True),
	maxEventsToPrint = cms.untracked.int32(0),

	crossSection = cms.untracked.double(-1.0),
	filterEfficiency = cms.untracked.double(1.0),

	HerwigParameters = cms.PSet(
		parameterSets = cms.vstring(
			'jimmyUESettings',
			'herwigHZZ4mu'
#			'herwigQCDjets'
		),
		jimmyUESettings = cms.vstring(
			'JMUEO      = 1          ! multiparton interaction model', 
			'PTJIM      = 4.449      ! 2.8x(sqrt(s)/1.8TeV)^0.27 @ 10 TeV', 
			'JMRAD(73)  = 1.8        ! inverse proton radius squared', 
			'PRSOF      = 0.0        ! prob. of a soft underlying event'
		),
		herwigHZZ4mu = cms.vstring(
			'RMASS(201) = 175        ! Mass of the Higgs boson',
			'IPROC      = 1611       ! Process gg -> H -> ZZ',
			'MODBOS(1)  = 3          ! enforde first Z -> mumu',
			'MODBOS(2)  = 3          ! enforce second Z -> mumu',
			'MODPDF(1)  = 20060      ! PDF set according to LHAGLUE',
			'MODPDF(2)  = 20060      ! MRST2001LO',
			'PTJIM	    = 2.5',
			'PTMIN      = 2.5'
		),
		herwigQCDjets = cms.vstring(
			'IPROC      = 2505      ! QCD 2->2 processes',
			'PTMIN      = 80.       ! minimum pt in hadronic jet',
			'MODPDF(1)  = 10041     ! PDF set according to LHAGLUE',
			'MODPDF(2)  = 10041     ! CTEQ6L',
		)
	)
)

process.output = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string('herwigHZZ4mu.root')
)
