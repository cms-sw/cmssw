import FWCore.ParameterSet.Config as cms

pfDeepFlavourJetTags = cms.EDProducer(
    'DeepFlavourJetTagProducer',
    src = cms.InputTag('DeepFlavourTagInfos'),
    outputs = cms.vstring(["probb"]),
)
