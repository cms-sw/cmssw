import FWCore.ParameterSet.Config as cms

pythia8CP5CR1TuneSettingsBlock = cms.PSet(
    pythia8CP5CR1TuneSettings = cms.vstring(
	'Tune:pp 14',
	'Tune:ee 7',
	'PDF:pSet=20',
	'MultipartonInteractions:alphaSvalue=0.118',
	'MultipartonInteractions:alphaSorder=2',
	'MultipartonInteractions:bProfile=2',
	'MultipartonInteractions:pT0Ref=1.375',
	'MultipartonInteractions:ecmPow=0.03283',
	'MultipartonInteractions:coreFraction=0.4446',
	'MultipartonInteractions:coreRadius=0.6046',
	'ColourReconnection:mode=1',
	'BeamRemnants:remnantMode=1',
	'ColourReconnection:junctionCorrection=0.238',
	'ColourReconnection:timeDilationPar=8.58',
	'ColourReconnection:m0=1.721',
	'StringZ:aLund=0.38',
	'StringZ:bLund=0.64',
	'StringFlav:probQQtoQ=0.078',
	'StringFlav:probStoUD=0.2',
	'SpaceShower:alphaSorder=2',
	'SpaceShower:alphaSvalue=0.118',
	'SigmaProcess:alphaSvalue=0.118',
	'SigmaProcess:alphaSorder=2',
	'TimeShower:alphaSorder=2',
	'TimeShower:alphaSvalue=0.118',
	'SigmaTotal:zeroAXB=off',
    )
)
