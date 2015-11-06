import FWCore.ParameterSet.Config as cms

herwigDefaultsBlock = cms.PSet(

	dataLocation = cms.string('${HERWIGPATH}'),
	repository = cms.string('HerwigDefaults.rpo'),


	eventHandlers = cms.string('/Herwig/EventHandlers'),
	generatorModule = cms.string('/Herwig/Generators/LHCGenerator'),
	run = cms.string('LHC'),

	hwpp_cmsDefaults = cms.vstring(
		'+hwpp_basicSetup',
		'+hwpp_setParticlesStableForDetector',
	),

	hwpp_basicSetup = cms.vstring(
		# /Herwig/Generators
		'create ThePEG::RandomEngineGlue /Herwig/RandomGlue',
		'set /Herwig/Generators/LHCGenerator:RandomNumberGenerator /Herwig/RandomGlue',
		'set /Herwig/Generators/LHCGenerator:NumberOfEvents 10000000',
		'set /Herwig/Generators/LHCGenerator:DebugLevel 1',
                'set /Herwig/Generators/LHCGenerator:UseStdout 0',		
		'set /Herwig/Generators/LHCGenerator:PrintEvent 0',
		'set /Herwig/Generators/LHCGenerator:MaxErrors 10000',
	),

	# Disable decays of particles with ctau > 10mm
	hwpp_setParticlesStableForDetector = cms.vstring(
		# /Herwig/Particles
		'set /Herwig/Particles/mu-:Stable Stable',
		'set /Herwig/Particles/mu+:Stable Stable',
		'set /Herwig/Particles/Sigma-:Stable Stable',
		'set /Herwig/Particles/Sigmabar+:Stable Stable',
		'set /Herwig/Particles/Lambda0:Stable Stable',
		'set /Herwig/Particles/Lambdabar0:Stable Stable',
		'set /Herwig/Particles/Sigma+:Stable Stable',
		'set /Herwig/Particles/Sigmabar-:Stable Stable',
		'set /Herwig/Particles/Xi-:Stable Stable',
		'set /Herwig/Particles/Xibar+:Stable Stable',
		'set /Herwig/Particles/Xi0:Stable Stable',
		'set /Herwig/Particles/Xibar0:Stable Stable',
		'set /Herwig/Particles/Omega-:Stable Stable',
		'set /Herwig/Particles/Omegabar+:Stable Stable',
		'set /Herwig/Particles/pi+:Stable Stable',
		'set /Herwig/Particles/pi-:Stable Stable',
		'set /Herwig/Particles/K+:Stable Stable',
		'set /Herwig/Particles/K-:Stable Stable',
		'set /Herwig/Particles/K_S0:Stable Stable',
		'set /Herwig/Particles/K_L0:Stable Stable',
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

