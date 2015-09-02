import FWCore.ParameterSet.Config as cms

pythia8CUEP8S1cteqSettingsBlock = cms.PSet(
    pythia8CUEP8S1cteqSettings = cms.vstring(
        'Tune:pp 5',
        'Tune:ee 3',
        'MultipartonInteractions:pT0Ref=2.1006',
        'MultipartonInteractions:ecmPow=0.21057',
        'MultipartonInteractions:expPow=1.6089',
        'MultipartonInteractions:a1=0.00',
	'ColourReconnection:range=3.31257',
   )
)
