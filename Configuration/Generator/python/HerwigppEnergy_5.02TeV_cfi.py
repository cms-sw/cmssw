import FWCore.ParameterSet.Config as cms

# Center-of-mass energy 5.02 TeV

herwigppEnergySettingsBlock = cms.PSet(

	hwpp_cm_5_02TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 5020.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.0*GeV',
	),
)
