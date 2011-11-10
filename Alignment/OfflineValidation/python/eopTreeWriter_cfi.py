import FWCore.ParameterSet.Config as cms

energyOverMomentumTree = cms.EDAnalyzer('EOP',
             src = cms.InputTag('TrackRefitter')
)
