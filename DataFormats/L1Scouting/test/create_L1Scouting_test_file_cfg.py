import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

process.l1ScoutingTestProducer = cms.EDProducer("TestWriteL1Scouting",
  bxValues = cms.vuint32(42, 512),
  muonValues = cms.vint32(1, 2, 3),
  jetValues = cms.vint32(4, 5, 6, 7),
  eGammaValues = cms.vint32(8, 9, 10),
  tauValues = cms.vint32(11, 12),
  bxSumsValues = cms.vint32(13),
  bmtfStubValues = cms.vint32(1, 2),
  caloTowerStubValues = cms.vint32(14, 15),

)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testL1Scouting.root')
)

process.path = cms.Path(process.l1ScoutingTestProducer)
process.endPath = cms.EndPath(process.out)
