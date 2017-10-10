import FWCore.ParameterSet.Config as cms

pythia8CP2SettingsBlock = cms.PSet(
    pythia8CP2Settings = cms.vstring(
	'Tune:pp 14',
	'Tune:ee 7',
	'MultipartonInteractions:ecmPow=0.1391',
	'PDF:pSet=17',
	'MultipartonInteractions:bProfile=2',
	'MultipartonInteractions:pT0Ref=2.306',
	'MultipartonInteractions:coreRadius=0.3755',
	'MultipartonInteractions:coreFraction=0.3269',
	'ColourReconnection:range=2.323',
	'SigmaTotal:zeroAXB=off', 
	'SpaceShower:rapidityOrder=off',
	'SpaceShower:alphaSvalue=0.13',
        'TimeShower:alphaSvalue=0.13',
	)
)

