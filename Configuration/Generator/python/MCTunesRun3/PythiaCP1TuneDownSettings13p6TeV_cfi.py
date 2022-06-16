import FWCore.ParameterSet.Config as cms

pythia8CP1TuneDownSettings13p6TeVBlock = cms.PSet(
    pythia8CP1TuneDownSettings13p6TeV = cms.vstring(
        'Tune:pp 14',
	'Tune:ee 7',
	'MultipartonInteractions:bProfile=2',
        'MultipartonInteractions:ecmPow=0.154',
        'MultipartonInteractions:pT0Ref=2.4',
        'MultipartonInteractions:coreRadius=0.5962',
        'MultipartonInteractions:coreFraction=0.3902',
        'ColourReconnection:range=8.5'
	'SigmaTotal:zeroAXB=off',
	'SpaceShower:rapidityOrder=off',
        'SigmaTotal:mode = 0',
        'SigmaTotal:sigmaEl = 22.08',
        'SigmaTotal:sigmaTot = 101.037',
        'PDF:pSet=LHAPDF6:NNPDF31_lo_as_0130',
	)
)
