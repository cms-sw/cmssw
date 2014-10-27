import FWCore.ParameterSet.Config as cms

pythia8CUEP8M1SettingsBlock = cms.PSet(
    pythia8CUEP8M1Settings = cms.vstring(
        'Tune:pp 14',
        'Tune:ee 7',
        'MultipartonInteractions:pT0Ref=2.4024',
        'MultipartonInteractions:ecmPow=0.25208',
        'MultipartonInteractions:expPow=1.6',
    )
)
