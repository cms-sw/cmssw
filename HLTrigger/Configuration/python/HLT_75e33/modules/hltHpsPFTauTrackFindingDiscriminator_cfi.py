import FWCore.ParameterSet.Config as cms

hltHpsPFTauTrackFindingDiscriminator = cms.EDProducer("PFRecoTauDiscriminationByLeadingObjectPtCut",
    MinPtLeadingObject = cms.double(0.0),
    PFTauProducer = cms.InputTag("hltHpsPFTauProducer"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string('and')
    ),
    UseOnlyChargedHadrons = cms.bool(True)
)
