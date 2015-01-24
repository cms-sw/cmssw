import FWCore.ParameterSet.Config as cms


generator = cms.EDFilter("Herwig6GeneratorFilter",
	comEnergy = cms.double(7000.0),
	crossSection = cms.untracked.double(4.92395e+07),
	doMPInteraction = cms.bool(True),
	emulatePythiaStatusCodes = cms.untracked.bool(True),
	filterEfficiency = cms.untracked.double(1.0),
	herwigHepMCVerbosity = cms.untracked.bool(False),
	herwigVerbosity = cms.untracked.int32(0),
	lhapdfSetPath = cms.untracked.string(''),
	maxEventsToPrint = cms.untracked.int32(0),
	printCards = cms.untracked.bool(False),
	useJimmy = cms.bool(True),

	HerwigParameters = cms.PSet(
		herwigQCDjets = cms.vstring(
			'IPROC     = 1500    ! QCD 2->2 processes', 
			'PTMIN     = 30.    ! minimum pt in hadronic jet',
			'MODPDF(1) = 10041   ! PDF set according to LHAGLUE', 
			'MODPDF(2) = 10041   ! CTEQ6L', 
			'JMUEO     = 1       ! multiparton interaction model',
            'PTJIM     = 4.040   ! 2.8x(sqrt(s)/1.8TeV)^0.27 @ 7 TeV', 
			'JMRAD(73) = 1.8     ! inverse proton radius squared',
			'PRSOF     = 0.0     ! prob. of a soft underlying event',
			'MAXER     = 1000000 ! max error'
		),
		parameterSets = cms.vstring('herwigQCDjets')
	),
)

ProductionFilterSequence = cms.Sequence(generator)
