import FWCore.ParameterSet.Config as cms

JPTTrackProducer = cms.EDProducer('JPTTrackProducer',
  ZSPCorrectedJetsTag = cms.InputTag('ZSPJetCorJetAntiKt7'),
  JPTCorrectorName = cms.string('JetPlusTrackZSPCorrectorAntiKt7'),
  JetIndex = cms.uint32(0),
  ProduceInCaloInVertex = cms.bool(True),
  ProduceOutCaloInVertex = cms.bool(True),
  ProduceInCaloOutVertex = cms.bool(True),
  ProducePions = cms.bool(True),
  ProduceMuons = cms.bool(False),
  ProduceElectrons = cms.bool(False)
)

