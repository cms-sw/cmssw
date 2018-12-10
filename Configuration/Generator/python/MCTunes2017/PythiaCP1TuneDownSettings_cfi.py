import FWCore.ParameterSet.Config as cms

pythia8CP1TuneDownSettingsBlock = cms.PSet(
    pythia8CP1TuneDownSettings = cms.vstring(
        'Tune:pp 14',
	'Tune:ee 7',
	'PDF:pSet=17',
	'MultipartonInteractions:bProfile=2',
        'MultipartonInteractions:ecmPow=0.154',
        'MultipartonInteractions:pT0Ref=2.4',
        'MultipartonInteractions:coreRadius=0.5962',
        'MultipartonInteractions:coreFraction=0.3902',
        'ColourReconnection:range=8.5'
	'SigmaTotal:zeroAXB=off',
	'SpaceShower:rapidityOrder=off',
	)
)
