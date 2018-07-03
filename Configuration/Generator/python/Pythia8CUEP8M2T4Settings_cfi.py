import FWCore.ParameterSet.Config as cms

pythia8CUEP8M2T4SettingsBlock = cms.PSet(
    pythia8CUEP8M2T4Settings = cms.vstring(
        'Tune:pp 14',
        'Tune:ee 7',
        'MultipartonInteractions:ecmPow=0.25208',
	    'SpaceShower:alphaSvalue=0.1108',
        'PDF:pSet=LHAPDF6:NNPDF30_lo_as_0130',
	    'MultipartonInteractions:pT0Ref=2.20e+00',
        'MultipartonInteractions:expPow=1.60e+00',
        'ColourReconnection:range=6.59e+00',
    )
)
