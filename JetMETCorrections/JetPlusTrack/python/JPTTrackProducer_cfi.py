import FWCore.ParameterSet.Config as cms

JPTTrackProducer = cms.EDProducer('JPTTrackProducer',
  ZSPCorrectedJetsTag = cms.InputTag('ZSPJetCorJetAntiKt7'),
  JPTCorrectorName = cms.string('JetPlusTrackZSPCorrectorAntiKt7')
)

