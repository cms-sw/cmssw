import FWCore.ParameterSet.Config as cms

hltHpsPFTauTrackPt1Discriminator = cms.EDProducer( "PFRecoTauDiscriminationByLeadingObjectPtCut",
    MinPtLeadingObject = cms.double( 1.0 ),
    Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
    UseOnlyChargedHadrons = cms.bool( True ),
    PFTauProducer = cms.InputTag( "hltHpsPFTauProducer" )
)
