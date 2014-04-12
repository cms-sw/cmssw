import FWCore.ParameterSet.Config as cms

herwigValidationBlock = cms.PSet(
	eventsToPrint = cms.untracked.uint32(1),
	dumpConfig  = cms.untracked.string("dump.config"),
	dumpEvents  = cms.untracked.string("dump.hepmc"),

	validationMSSM = cms.vstring(
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

	validationQCD = cms.vstring(
		'cd /Herwig/MatrixElements/',
		'insert SimpleQCD:MatrixElements[0] MEQCD2to2',
#		'insert SimpleQCD:Reweights[0] /Herwig/Weights/reweightMinPT',
		'cd /',
		'set /Herwig/Cuts/JetKtCut:MinKT 50*GeV',
		'set /Herwig/Cuts/JetKtCut:MaxKT 100*GeV',
		'set /Herwig/UnderlyingEvent/MPIHandler:IdenticalToUE 0',
	)
)
