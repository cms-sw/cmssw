import FWCore.ParameterSet.Config as cms

hltFixedGridRhoProducerFastjetAllTau = cms.EDProducer( "FixedGridRhoProducerFastjet",
    pfCandidatesTag = cms.InputTag( "particleFlowTmp" ),
    maxRapidity = cms.double( 5.0 ),
    gridSpacing = cms.double( 0.55 )
)
