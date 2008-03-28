import FWCore.ParameterSet.Config as cms

# configuration of the RecoAnalyzer
#
# makes some plots of the reconstruction
RecoAnalyzer = cms.EDAnalyzer("RecoAnalyzer",
    SearchWindowPhiTEC = cms.untracked.double(0.05),
    # list of digi producers
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('\0'),
        DigiProducer = cms.string('siStripDigis')
    )),
    SearchWindowPhiTIB = cms.untracked.double(0.05),
    ROOTFileName = cms.untracked.string('RecoAnalyzer.histos.root'),
    SearchWindowZTOB = cms.untracked.double(1.0),
    ROOTFileCompression = cms.untracked.int32(1),
    ClusterProducer = cms.string('siStripClusters'),
    SearchWindowZTIB = cms.untracked.double(1.0),
    RecHitProducer = cms.string('siStripMatchedRecHits'),
    SearchWindowPhiTOB = cms.untracked.double(0.05)
)


