import FWCore.ParameterSet.Config as cms

pythia8CUETP8M1DownVariationSettingsBlock = cms.PSet(
    pythia8CUETP8M1DownVariationSettings = cms.vstring(
        'Tune:pp 14',
        'Tune:ee 7',
        'MultipartonInteractions:pT0Ref=2.60468e+00',
        'MultipartonInteractions:ecmPow=2.5208e-01',
        'MultipartonInteractions:expPow=1.515108e+00',
        'ColourReconnection:range=4.200868e+00',
    )
)
