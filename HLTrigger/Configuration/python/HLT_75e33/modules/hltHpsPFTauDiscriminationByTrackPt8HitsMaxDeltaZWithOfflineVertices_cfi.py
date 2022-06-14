import FWCore.ParameterSet.Config as cms

hltHpsPFTauDiscriminationByTrackPt8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("PFRecoTauDiscriminationByLeadingObjectPtCut",
    MinPtLeadingObject = cms.double(1.0),
    PFTauProducer = cms.InputTag("hltHpsPFTaus8HitsMaxDeltaZWithOfflineVertices"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string('and')
    ),
    UseOnlyChargedHadrons = cms.bool(True)
)
