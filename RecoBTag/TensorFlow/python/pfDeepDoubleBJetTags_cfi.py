import FWCore.ParameterSet.Config as cms

pfDeepDoubleBJetTags = cms.EDProducer('DeepDoubleXTFJetTagsProducer',
  src = cms.InputTag('pfDeepDoubleXTagInfosNopt')
)
