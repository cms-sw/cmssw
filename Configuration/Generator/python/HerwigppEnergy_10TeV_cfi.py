import FWCore.ParameterSet.Config as cms

# Center-of-mass energy 10 TeV

herwigEnergySettingsBlock = cms.PSet(

	cm10TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 10000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.1*GeV',
	),
)

