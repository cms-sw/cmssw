import FWCore.ParameterSet.Config as cms

process = cms.Process('Analysis')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:Exhume.root")
)

process.analyzer = cms.EDAnalyzer("ExhumeAnalyzer",
        GenParticleTag = cms.InputTag("genParticles"),
        EBeam = cms.double(4000.)
)

process.add_(cms.Service("TFileService",
		fileName = cms.string("analysisExhume_histos.root")
	)
)

process.analysis = cms.Path(process.analyzer)
