import FWCore.ParameterSet.Config as cms

pythia8CP2TuneDownSettingsBlock = cms.PSet(
    pythia8CP2TuneDownSettings = cms.vstring(
	'Tune:pp 14',
	'Tune:ee 7',
	'PDF:pSet=17',
	'MultipartonInteractions:bProfile=2',
	'MultipartonInteractions:ecmPow=0.1391',
	'MultipartonInteractions:pT0Ref=2.333',
	'MultipartonInteractions:coreRadius=0.3438',
	'MultipartonInteractions:coreFraction=0.2301',
	'ColourReconnection:range=2.563',
	'SigmaTotal:zeroAXB=off', 
	'SpaceShower:rapidityOrder=off',
	'SpaceShower:alphaSvalue=0.13',
        'TimeShower:alphaSvalue=0.13',
	)
)

