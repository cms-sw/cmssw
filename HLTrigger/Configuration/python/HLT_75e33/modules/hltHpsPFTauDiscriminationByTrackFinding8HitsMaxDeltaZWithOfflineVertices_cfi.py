import FWCore.ParameterSet.Config as cms

hltHpsPFTauDiscriminationByTrackFinding8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("PFRecoTauDiscriminationByLeadingObjectPtCut",
    MinPtLeadingObject = cms.double(0.0),
    PFTauProducer = cms.InputTag("hltHpsPFTaus8HitsMaxDeltaZWithOfflineVertices"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string('and')
    ),
    UseOnlyChargedHadrons = cms.bool(True)
)
