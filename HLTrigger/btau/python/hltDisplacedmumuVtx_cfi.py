import FWCore.ParameterSet.Config as cms

hltDisplacedmumuVtx = cms.EDProducer("HLTDisplacedmumuVtxProducer",
    Src = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag("" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinPtPair = cms.double( 0.0 ),
    MinInvMass = cms.double( 1 ),
    MaxInvMass = cms.double( 20 ),
    ChargeOpt = cms.int32( -1 ),
)
