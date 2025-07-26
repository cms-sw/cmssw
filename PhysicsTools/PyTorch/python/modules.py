import FWCore.ParameterSet.Config as cms


DataProducerStruct = cms.EDProducer('DataProducer@alpaka',
  batchSize = cms.uint32(32),
  alpaka = cms.untracked.PSet(),
)

JitClassificationProducerStruct = cms.EDProducer('JitClassificationProducer@alpaka',
  inputs = cms.InputTag('DataProducerStruct'),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

JitRegressionProducerStruct = cms.EDProducer('JitRegressionProducer@alpaka',
  inputs = cms.InputTag('DataProducerStruct'),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

AotClassificationProducerStruct = cms.EDProducer('AotClassificationProducer@alpaka',
  inputs = cms.InputTag('DataProducerStruct'),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

AotRegressionProducerStruct = cms.EDProducer('AotRegressionProducer@alpaka',
  inputs = cms.InputTag('DataProducerStruct'),
  modelPath = cms.FileInPath('model.pt'),
  alpaka = cms.untracked.PSet(),
)

CombinatoricsProducerStruct = cms.EDProducer('CombinatoricsProducer@alpaka',
  inputs = cms.InputTag('DataProducerStruct'),
  alpaka = cms.untracked.PSet(),
)
