import FWCore.ParameterSet.Config as cms
from GeneratorInterface.ThePEGInterface.herwigDefaults_cff import *

configurationMetadata = cms.untracked.PSet(
	version = cms.untracked.string('$Revision: 1.1 $'),
	name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/GeneratorInterface/ThePEGInterface/test/testThePEGProducer.py,v $'),
	annotation = cms.untracked.string('Herwig++ example - QCD events, MRST2001 used, MinKT=1400 GeV')
)

source = cms.Source("EmptySource")

generator = cms.EDProducer("ThePEGGeneratorFilter",
	herwigDefaultsBlock,

	eventsToPrint = cms.untracked.uint32(1),
	dumpConfig  = cms.untracked.string(""),
	dumpEvents  = cms.untracked.string(""),

	configFiles = cms.vstring(
#		'MSSM.model'
	),

	parameterSets = cms.vstring(
		'cmsDefaults', 
#		'mssm',
		'validation'
	),

	mssm = cms.vstring(
		'cd /Herwig/NewPhysics', 
		'set HPConstructor:IncludeEW No', 
		'set TwoBodyDC:CreateDecayModes No', 
		'setup MSSM/Model ${HERWIGPATH}/SPhenoSPS1a.spc', 
		'insert NewModel:DecayParticles 0 /Herwig/Particles/~d_L', 
		'insert NewModel:DecayParticles 1 /Herwig/Particles/~u_L', 
		'insert NewModel:DecayParticles 2 /Herwig/Particles/~e_R-', 
		'insert NewModel:DecayParticles 3 /Herwig/Particles/~mu_R-', 
		'insert NewModel:DecayParticles 4 /Herwig/Particles/~chi_10', 
		'insert NewModel:DecayParticles 5 /Herwig/Particles/~chi_20', 
		'insert NewModel:DecayParticles 6 /Herwig/Particles/~chi_2+'
	),

	validation = cms.vstring(
		'cd /Herwig/MatrixElements/', 
		'insert SimpleQCD:MatrixElements[0] MEQCD2to2', 
		'set /Herwig/Cuts/QCDCuts:MHatMin 20.*GeV'
	)
)

ProducerSourceSequence = cms.Sequence(generator)
