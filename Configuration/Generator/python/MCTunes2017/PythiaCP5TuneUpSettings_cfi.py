import FWCore.ParameterSet.Config as cms

pythia8CP5TuneUpSettingsBlock = cms.PSet(
    pythia8CP5TuneUpSettings = cms.vstring(
    'Tune:pp 14',
        'Tune:ee 7',
        'MultipartonInteractions:ecmPow=0.03344',
        'PDF:pSet=20',
        'MultipartonInteractions:bProfile=2',
        'MultipartonInteractions:pT0Ref=1.407',
        'MultipartonInteractions:coreRadius=0.6671',
        'MultipartonInteractions:coreFraction=0.4281',
        'ColourReconnection:range=4.881',
        'SigmaTotal:zeroAXB=off',
        'SpaceShower:alphaSorder=2',
        'SpaceShower:alphaSvalue=0.118',
        'SigmaProcess:alphaSvalue=0.118',
        'SigmaProcess:alphaSorder=2',
        'MultipartonInteractions:alphaSvalue=0.118',
        'MultipartonInteractions:alphaSorder=2',
        'TimeShower:alphaSorder=2',
        'TimeShower:alphaSvalue=0.118',
        )
)
