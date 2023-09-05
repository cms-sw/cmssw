import FWCore.ParameterSet.Config as cms

hltTauPFJets08Region = cms.EDProducer( "RecoTauJetRegionProducer",
    src = cms.InputTag( "hltAK4PFJets" ),
    deltaR = cms.double( 0.8 ),
    pfCandAssocMapSrc = cms.InputTag( "" ),
    verbosity = cms.int32( 0 ),
    maxJetAbsEta = cms.double( 99.0 ),
    minJetPt = cms.double( -1.0 ),
    pfCandSrc = cms.InputTag( "particleFlowTmp" )
)
