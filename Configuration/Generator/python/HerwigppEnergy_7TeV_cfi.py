import FWCore.ParameterSet.Config as cms

# Center-of-mass energy 7 TeV

herwigppEnergySettingsBlock = cms.PSet(

	hwpp_cm_7TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 7000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.0*GeV',
	),
)

