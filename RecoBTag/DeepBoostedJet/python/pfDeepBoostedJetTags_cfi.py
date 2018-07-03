import FWCore.ParameterSet.Config as cms

_pfDeepBoostedJetTags = cms.EDProducer('DeepBoostedJetTagsProducer',
  src = cms.InputTag('pfDeepBoostedJetTagInfos'),
  preprocessParams = cms.PSet(),
  model_path = cms.FileInPath('RecoBTag/Combined/data/DeepBoostedJet/V01/full/resnet-symbol.json'),
  param_path = cms.FileInPath('RecoBTag/Combined/data/DeepBoostedJet/V01/full/resnet-0000.params'),
  flav_table = cms.PSet(
    probTbcq = cms.vuint32(0),
    probTbqq = cms.vuint32(1),
    probTbc = cms.vuint32(2),
    probTbq = cms.vuint32(3),
    probWcq = cms.vuint32(4),
    probWqq = cms.vuint32(5),
    probZbb = cms.vuint32(6),
    probZcc = cms.vuint32(7),
    probZqq = cms.vuint32(8),
    probHbb = cms.vuint32(9),
    probHcc = cms.vuint32(10),
    probHqqqq = cms.vuint32(11),
    probQCDbb = cms.vuint32(12),
    probQCDcc = cms.vuint32(13),
    probQCDb = cms.vuint32(14),
    probQCDc = cms.vuint32(15),
    probQCDothers = cms.vuint32(16)
  )
)
