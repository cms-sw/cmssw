import FWCore.ParameterSet.Config as cms

hltHpsPFTauTrackFindingDiscriminator = cms.EDProducer( "PFRecoTauDiscriminationByLeadingObjectPtCut",
    MinPtLeadingObject = cms.double( 0.0 ),
    Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
    UseOnlyChargedHadrons = cms.bool( True ),
    PFTauProducer = cms.InputTag( "hltHpsPFTauProducer" )
)
# foo bar baz
# J2yuWbrlg2R9e
# 5HWF9vWH1mM12
