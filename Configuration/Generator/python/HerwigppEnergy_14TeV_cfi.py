import FWCore.ParameterSet.Config as cms

# Center-of-mass energy 14 TeV

herwigppEnergySettingsBlock = cms.PSet(

	hwpp_cm_14TeV = cms.vstring(
		'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 14000.0',
		'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.2*GeV',
	),
)

