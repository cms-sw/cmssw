import FWCore.ParameterSet.Config as cms

hltHPDFilter= cms.EDFilter( "HLTHPDFilter",
    inputTag = cms.InputTag( "Hbhereco" ),
    energy = cms.double( -999.0 ),
    hpdSpikeEnergy = cms.double( 10.1 ),
    hpdSpikeIsolationEnergy = cms.double( 1.1 ),
    rbxSpikeEnergy = cms.double( 40.0 ),
    rbxSpikeUnbalance = cms.double( 0.21 )
)

