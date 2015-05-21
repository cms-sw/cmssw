import FWCore.ParameterSet.Config as cms

herwigDefaultsBlock = cms.PSet(

	dataLocation = cms.string('${HERWIGPATH}'),
	repository = cms.string('HerwigDefaults.rpo'),


	eventHandlers = cms.string('/Herwig/EventHandlers'),
	generatorModule = cms.string('/Herwig/Generators/LHCGenerator'),
	run = cms.string('LHC'),

	cmsDefaults = cms.vstring(
		'+basicSetup',
		'+setParticlesStableForDetector',
	),

	basicSetup = cms.vstring(
		'cd /Herwig/Generators',
		'create ThePEG::RandomEngineGlue /Herwig/RandomGlue',
		'set LHCGenerator:RandomNumberGenerator /Herwig/RandomGlue',
		'set LHCGenerator:NumberOfEvents 10000000',
		'set LHCGenerator:DebugLevel 1',
                'set LHCGenerator:UseStdout 0',		
		'set LHCGenerator:PrintEvent 0',
		'set LHCGenerator:MaxErrors 10000',
		'cd /',
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

	# PDF presets
	# Can be found under HerwigppPDF_
	

	# Center-of-mass energy presets
	# Can be found under HerwigppEnergy_


	# reweight presets
	# Can be found under HerwigppReweight_



	# Default settings for using LHE files
	# Can be found under HerwigppLHEFile_

)

