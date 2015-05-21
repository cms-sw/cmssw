import FWCore.ParameterSet.Config as cms

# Center-of-mass energy 8 TeV

herwigEnergySettingsBlock = cms.PSet(

	cm8TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 8000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.0*GeV',
	),
)

