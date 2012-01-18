import FWCore.ParameterSet.Config as cms

herwigDefaultsBlock = cms.PSet(
	dataLocation = cms.string('${HERWIGPATH}'),

	repository = cms.string('HerwigDefaults.rpo'),
	eventHandlers = cms.string('/Herwig/EventHandlers'),
	generatorModule = cms.string('/Herwig/Generators/LHCGenerator'),

	run = cms.string('LHC'),

	cmsDefaults = cms.vstring(
		'+pdfMRST2001',
		'+cm14TeV',
		'+ue_2_3',
		'+basicSetup',
		'+setParticlesStableForDetector',
	),

	basicSetup = cms.vstring(
		'cd /Herwig/Generators',
		'create ThePEG::RandomEngineGlue /Herwig/RandomGlue',
		'set LHCGenerator:RandomNumberGenerator /Herwig/RandomGlue',
		'set LHCGenerator:NumberOfEvents 10000000',
		'set LHCGenerator:DebugLevel 1',
		'set LHCGenerator:PrintEvent 0',
		'set LHCGenerator:MaxErrors 10000',
		'cd /Herwig/Particles',
		'set p+:PDF /Herwig/Partons/cmsPDFSet',
		'set pbar-:PDF /Herwig/Partons/cmsPDFSet',
		'set K0:Width 1e300*GeV',
		'set Kbar0:Width 1e300*GeV',
		'cd /',
	),

	# PDF presets
	##############################

	# Default pdf for Herwig++ 2.3
	pdfMRST2001 = cms.vstring(
		'cd /Herwig/Partons',
		'create Herwig::MRST MRST2001 HwMRST.so',
		'setup MRST2001 ${HERWIGPATH}/PDF/mrst/2001/lo2002.dat',
		'set MRST2001:RemnantHandler HadronRemnants',
		'cp MRST2001 cmsPDFSet',
		'cd /',
	),
	# Default pdf for Herwig++ 2.4
	pdfMRST2008LOss = cms.vstring(
		'cp /Herwig/Partons/MRST /Herwig/Partons/cmsPDFSet',
	),
	pdfCTEQ5L = cms.vstring(
		'cd /Herwig/Partons',
		'create ThePEG::LHAPDF CTEQ5L ThePEGLHAPDF.so',
		'set CTEQ5L:PDFName cteq5l.LHgrid',
		'set CTEQ5L:RemnantHandler HadronRemnants',
		'cp CTEQ5L cmsPDFSet',
		'cd /',
	),
	pdfCTEQ6L1 = cms.vstring(
		'cd /Herwig/Partons',
		'create ThePEG::LHAPDF CTEQ6L1 ThePEGLHAPDF.so',
		'set CTEQ6L1:PDFName cteq6ll.LHpdf',
		'set CTEQ6L1:RemnantHandler HadronRemnants',
		'cp CTEQ6L1 cmsPDFSet',
		'cd /',
	),

	# CME presets
	##############################

	cm7TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 7000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.0*GeV',
	),
	cm8TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 8000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.0*GeV',
	),
	cm10TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 10000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.1*GeV',
	),
	cm14TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 14000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.2*GeV',
	),

	# UE tunes
	##############################

	# UE Tune from Herwig++ 2.3 (MRST2001)
	ue_2_3 = cms.vstring(
		'cd /Herwig/UnderlyingEvent',
		'set KtCut:MinKT 4.0',
		'set UECuts:MHatMin 8.0',
		'set MPIHandler:InvRadius 1.5',
		'cd /',
	),
	# UE Tune from Herwig++ 2.4 (MRST2008LO**)
	ue_2_4 = cms.vstring(
		'cd /Herwig/UnderlyingEvent',
		'set KtCut:MinKT 4.3',
		'set UECuts:MHatMin 8.6',
		'set MPIHandler:InvRadius 1.2',
		'cd /',
	),

	# reweight presets
	##############################

	reweightConstant = cms.vstring(
		'mkdir /Herwig/Weights',
		'cd /Herwig/Weights',
		'create ThePEG::ReweightConstant reweightConstant ReweightConstant.so',
		'cd /',
		'set /Herwig/Weights/reweightConstant:C 1',
		'insert SimpleQCD:Reweights[0] /Herwig/Weights/reweightConstant',
	),
	reweightPthat = cms.vstring(
		'mkdir /Herwig/Weights',
		'cd /Herwig/Weights',
		'create ThePEG::ReweightMinPT reweightMinPT ReweightMinPT.so',
		'cd /',
		'set /Herwig/Weights/reweightMinPT:Power 4.5',
		'set /Herwig/Weights/reweightMinPT:Scale 15*GeV',
		'insert SimpleQCD:Reweights[0] /Herwig/Weights/reweightMinPT',
	),

	# Disable decays of particles with ctau > 10mm
	setParticlesStableForDetector = cms.vstring(
		'cd /Herwig/Particles',
		'set mu-:Stable Stable',
		'set mu+:Stable Stable',
		'set Sigma-:Stable Stable',
		'set Sigmabar+:Stable Stable',
		'set Lambda0:Stable Stable',
		'set Lambdabar0:Stable Stable',
		'set Sigma+:Stable Stable',
		'set Sigmabar-:Stable Stable',
		'set Xi-:Stable Stable',
		'set Xibar+:Stable Stable',
		'set Xi0:Stable Stable',
		'set Xibar0:Stable Stable',
		'set Omega-:Stable Stable',
		'set Omegabar+:Stable Stable',
		'set pi+:Stable Stable',
		'set pi-:Stable Stable',
		'set K+:Stable Stable',
		'set K-:Stable Stable',
		'set K_S0:Stable Stable',
		'set K_L0:Stable Stable',
		'cd /',
	),

	# Default settings for using LHE files
	lheDefaults = cms.vstring(
		'cd /Herwig/Cuts',
		'create ThePEG::Cuts NoCuts',
		'cd /Herwig/EventHandlers',
		'create ThePEG::LesHouchesInterface LHEReader',
		'set LHEReader:Cuts /Herwig/Cuts/NoCuts',
		'create ThePEG::LesHouchesEventHandler LHEHandler',
		'set LHEHandler:WeightOption VarWeight',
		'set LHEHandler:PartonExtractor /Herwig/Partons/QCDExtractor',
		'set LHEHandler:CascadeHandler /Herwig/Shower/ShowerHandler',
		'set LHEHandler:HadronizationHandler /Herwig/Hadronization/ClusterHadHandler',
		'set LHEHandler:DecayHandler /Herwig/Decays/DecayHandler',
		'insert LHEHandler:LesHouchesReaders 0 LHEReader',
		'cd /Herwig/Generators',
		'set LHCGenerator:EventHandler /Herwig/EventHandlers/LHEHandler',
		'cd /Herwig/Shower',
		'set Evolver:HardVetoScaleSource Read',
		'set Evolver:MECorrMode No',
		'cd /',
	),
	lheDefaultPDFs = cms.vstring(
		'cd /Herwig/EventHandlers',
		'set LHEReader:PDFA /cmsPDFSet',
		'set LHEReader:PDFB /cmsPDFSet',
		'cd /',
	),

	# Default settings for using POWHEG
	powhegDefaults = cms.vstring(
		# Need to use an NLO PDF
		'cp /Herwig/Partons/MRST-NLO /cmsPDFSet',
		'set /Herwig/Particles/p+:PDF    /Herwig/Partons/MRST-NLO',
		'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/MRST-NLO',
		# and strong coupling
		'create Herwig::O2AlphaS O2AlphaS',
		'set /Herwig/Generators/LHCGenerator:StandardModelParameters:QCD/RunningAlphaS O2AlphaS',
		# Setup the POWHEG shower
		'cd /Herwig/Shower',
		# use the general recon for now
		'set KinematicsReconstructor:ReconstructionOption General',
		# create the Powheg evolver and use it instead of the default one
		'create Herwig::PowhegEvolver PowhegEvolver HwPowhegShower.so',
		'set ShowerHandler:Evolver PowhegEvolver',
		'set PowhegEvolver:ShowerModel ShowerModel',
		'set PowhegEvolver:SplittingGenerator SplittingGenerator',
		'set PowhegEvolver:MECorrMode 0',
		# create and use the Drell-yan hard emission generator
		'create Herwig::DrellYanHardGenerator DrellYanHardGenerator',
		'set DrellYanHardGenerator:ShowerAlpha AlphaQCD',
		'insert PowhegEvolver:HardGenerator 0 DrellYanHardGenerator',
		# create and use the gg->H hard emission generator
		'create Herwig::GGtoHHardGenerator GGtoHHardGenerator',
		'set GGtoHHardGenerator:ShowerAlpha AlphaQCD',
		'insert PowhegEvolver:HardGenerator 0 GGtoHHardGenerator',
	)
)
