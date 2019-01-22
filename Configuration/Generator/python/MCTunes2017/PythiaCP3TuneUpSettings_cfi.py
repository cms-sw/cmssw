import FWCore.ParameterSet.Config as cms

pythia8CP3TuneUpSettingsBlock = cms.PSet(
    pythia8CP3TuneUpSettings = cms.vstring(
    'Tune:pp 14',
    'Tune:ee 7',	
    'PDF:pSet=19',
    'MultipartonInteractions:bProfile=2',
    'MultipartonInteractions:ecmPow=0.02266',
    'MultipartonInteractions:pT0Ref=1.478',
    'MultipartonInteractions:coreRadius=0.4939',
    'MultipartonInteractions:coreFraction=0.3526',
    'ColourReconnection:range=8.154',
    'SigmaTotal:zeroAXB=off',    
    'SpaceShower:alphaSorder=2',
    'SpaceShower:alphaSvalue=0.118',
    'SigmaProcess:alphaSvalue=0.118',
    'SigmaProcess:alphaSorder=2',
    'MultipartonInteractions:alphaSvalue=0.118',
    'MultipartonInteractions:alphaSorder=2',
    'TimeShower:alphaSorder=2',
    'TimeShower:alphaSvalue=0.118',
    'SpaceShower:rapidityOrder=off', 
        )
)

