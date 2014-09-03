import FWCore.ParameterSet.Config as cms

pythia8CUEP8S1cteqSettingsBlock = cms.PSet(
    pythia8CUEP8S1cteqSettings = cms.vstring(
        'Tune:pp 5',
        'Tune:ee 3',
        'MultipleInteractions:pT0Ref=2.1006',
        'MultipleInteractions:ecmPow=0.21057',
        'MultipleInteractions:expPow=1.6089',
	'BeamRemnants:reconnectRange=3.31257',
   )
)

