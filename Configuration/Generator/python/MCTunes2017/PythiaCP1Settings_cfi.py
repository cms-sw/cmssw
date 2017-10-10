import FWCore.ParameterSet.Config as cms

pythia8CP1SettingsBlock = cms.PSet(
    pythia8CP1Settings = cms.vstring(
        'Tune:pp 14',
	'Tune:ee 7',
	'PDF:pSet=17',
	'MultipartonInteractions:bProfile=2',
	'MultipartonInteractions:ecmPow=0.1543',
	'MultipartonInteractions:pT0Ref=2.40',
	'MultipartonInteractions:coreRadius=0.5436',
	'MultipartonInteractions:coreFraction=0.6836',
	'ColourReconnection:range=2.633',
	'SigmaTotal:zeroAXB=off',
	'SpaceShower:rapidityOrder=off',
	)
)
