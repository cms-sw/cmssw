import FWCore.ParameterSet.Config as cms

process = cms.Process('DeepFlavourExporter')

process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
      "file:test_deep_flavour_MINIAODSIM.root"
    )
)

process.TFileService = cms.Service('TFileService',
        fileName = cms.string('output.root'),
        loseFileFast = cms.untracked.bool(True)
)

process.load('RecoBTag.DeepFlavour.DeepFlavourExporter_cfi')

process.seq = cms.Sequence(
    process.pfDeepFlavourExporter
)

process.p = cms.Path(process.seq)
