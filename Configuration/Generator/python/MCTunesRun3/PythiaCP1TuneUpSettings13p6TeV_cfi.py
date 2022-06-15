import FWCore.ParameterSet.Config as cms

pythia8CP1TuneUpSettings13p6TeVBlock = cms.PSet(
    pythia8CP1TuneUpSettings13p6TeV = cms.vstring(
        'Tune:pp 14',
	'Tune:ee 7',
	'MultipartonInteractions:bProfile=2',
        'MultipartonInteractions:ecmPow=0.154',
        'MultipartonInteractions:pT0Ref=2.3',
        'MultipartonInteractions:coreRadius=0.5832',
        'MultipartonInteractions:coreFraction=0.5064',
        'ColourReconnection:range=8.305',
	'SigmaTotal:zeroAXB=off',
	'SpaceShower:rapidityOrder=off',
        'SigmaTotal:mode = 0',
        'SigmaTotal:sigmaEl = 22.08',
        'SigmaTotal:sigmaTot = 101.037',
        'PDF:pSet=LHAPDF6:NNPDF31_lo_as_0130',
	)
)
