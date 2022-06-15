import FWCore.ParameterSet.Config as cms

pythia8CP2TuneDownSettings13p6TeVBlock = cms.PSet(
    pythia8CP2TuneDownSettings13p6TeV = cms.vstring(
	'Tune:pp 14',
	'Tune:ee 7',
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
        'SigmaTotal:mode = 0',
        'SigmaTotal:sigmaEl = 22.08',
        'SigmaTotal:sigmaTot = 101.037',
        'PDF:pSet=LHAPDF6:NNPDF31_lo_as_0130',
	)
)

