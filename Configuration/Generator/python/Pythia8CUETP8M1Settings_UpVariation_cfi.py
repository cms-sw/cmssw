import FWCore.ParameterSet.Config as cms

pythia8CUETP8M1UpVariationSettingsBlock = cms.PSet(
    pythia8CUETP8M1UpVariationSettings = cms.vstring(
        'Tune:pp 14',
        'Tune:ee 7',
        'MultipartonInteractions:pT0Ref=1.8238e+00',
        'MultipartonInteractions:ecmPow=2.5208e-01',
        'MultipartonInteractions:expPow=3.230749e+00',
        'ColourReconnection:range=7.600778e+00',
    )
)
