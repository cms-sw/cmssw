import FWCore.ParameterSet.Config as cms

mon = cms.EDAnalyzer("LaserDQM",
    SearchWindowPhiTEC = cms.untracked.double(0.05),
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('\0'),
        DigiProducer = cms.string('siStripDigis')
    )),
    SearchWindowPhiTIB = cms.untracked.double(0.05),
    SearchWindowZTOB = cms.untracked.double(1.0),
    DebugLevel = cms.untracked.int32(3),
    DQMFileName = cms.untracked.string('testDQM.root'),
    SearchWindowZTIB = cms.untracked.double(1.0),
    SearchWindowPhiTOB = cms.untracked.double(0.05)
)


