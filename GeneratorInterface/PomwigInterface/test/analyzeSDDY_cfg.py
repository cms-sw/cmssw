import FWCore.ParameterSet.Config as cms

process = cms.Process('Analysis')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring("rfio:/castor/cern.ch/user/a/antoniov/Summer08/POMWIG_SingleDiffractiveZmumuMinus_10TeV_M40_cff_py_GEN.root")
)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('PhysicsTools.HepMCCandAlgos.genParticles_cfi')

process.genParticles.abortOnUnknownPDGCode = False

process.SDDY = cms.EDAnalyzer("SDDYAnalyzer",
	GenParticleTag = cms.InputTag("genParticles"),
	Particle1Id = cms.int32(13),
	Particle2Id = cms.int32(13),
	debug = cms.untracked.bool(True)
)

process.add_(cms.Service("TFileService",
		fileName = cms.string("analysisSDZmumuMinus_histos.root")
	)
)

process.analysis = cms.Path(process.genParticles*process.SDDY)
