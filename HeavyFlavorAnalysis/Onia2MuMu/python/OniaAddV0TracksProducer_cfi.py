import FWCore.ParameterSet.Config as cms

oniaV0Tracks = cms.EDProducer('OniaAddV0TracksProducer',
   KShortTag = cms.InputTag("generalV0Candidates","Kshort"),
   LambdaTag = cms.InputTag("generalV0Candidates","Lambda")
)
