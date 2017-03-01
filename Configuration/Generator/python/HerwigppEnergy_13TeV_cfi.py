import FWCore.ParameterSet.Config as cms

# Center-of-mass energy 13 TeV

herwigppEnergySettingsBlock = cms.PSet(

        hwpp_cm_13TeV = cms.vstring(
                'set /Herwig/Generators/LHCGenerator:EventHandler:LuminosityFunction:Energy 13000.0',
                'set /Herwig/Shower/Evolver:IntrinsicPtGaussian 2.2*GeV',
        ), 
)

