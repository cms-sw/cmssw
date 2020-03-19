import FWCore.ParameterSet.Config as cms

pythia8CP5CR2TuneSettingsBlock = cms.PSet(
    pythia8CP5CR2TuneSettings = cms.vstring(
	'Tune:pp 14',
	'Tune:ee 7',
	'PDF:pSet=20',
	'MultipartonInteractions:bProfile=2',
	'MultipartonInteractions:pT0Ref=1.454',
	'MultipartonInteractions:ecmPow=0.0555',
	'MultipartonInteractions:coreFraction=0.4392',
	'MultipartonInteractions:coreRadius=0.6532',
	'ColourReconnection:mode=2',
	'ColourReconnection:m2Lambda=4.395',
	'ColourReconnection:fracGluon=0.9896',
	'SigmaTotal:zeroAXB=off',
	'SpaceShower:alphaSorder=2',
	'SpaceShower:alphaSvalue=0.118',
	'SigmaProcess:alphaSvalue=0.118',
	'SigmaProcess:alphaSorder=2',
	'MultipartonInteractions:alphaSvalue=0.118',
	'MultipartonInteractions:alphaSorder=2',
	'TimeShower:alphaSorder=2',
	'TimeShower:alphaSvalue=0.118',
    )
)
