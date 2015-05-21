import FWCore.ParameterSet.Config as cms

# Read in a LHE file from Powheg

herwigLHEFileSettingsBlock = cms.PSet(

	LHEFilePowheg = cms.vstring(
		'# Need to use an NLO PDF',
		'set /Herwig/Particles/p+:PDF    /Herwig/Partons/MRST-NLO',
		'set /Herwig/Particles/pbar-:PDF /Herwig/Partons/MRST-NLO',
		'# and strong coupling',
		'create Herwig::O2AlphaS O2AlphaS',
		'set /Herwig/Generators/LHCGenerator:StandardModelParameters:QCD/RunningAlphaS O2AlphaS',
		'# Setup the POWHEG shower',
		'cd /Herwig/Shower',
		'# use the general recon for now',
		'set KinematicsReconstructor:ReconstructionOption General',
		'# create the Powheg evolver and use it instead of the default one',
		'create Herwig::PowhegEvolver PowhegEvolver HwPowhegShower.so',
		'set ShowerHandler:Evolver PowhegEvolver',
		'set PowhegEvolver:ShowerModel ShowerModel',
		'set PowhegEvolver:SplittingGenerator SplittingGenerator',
		'set PowhegEvolver:MECorrMode 0',
		'# create and use the Drell-yan hard emission generator',
		'create Herwig::DrellYanHardGenerator DrellYanHardGenerator',
		'set DrellYanHardGenerator:ShowerAlpha AlphaQCD',
		'insert PowhegEvolver:HardGenerator 0 DrellYanHardGenerator',
		'# create and use the gg->H hard emission generator',
		'create Herwig::GGtoHHardGenerator GGtoHHardGenerator',
		'set GGtoHHardGenerator:ShowerAlpha AlphaQCD',
		'insert PowhegEvolver:HardGenerator 0 GGtoHHardGenerator',
		'cd /',
	),
)

