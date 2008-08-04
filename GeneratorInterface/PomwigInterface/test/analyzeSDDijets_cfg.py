import FWCore.ParameterSet.Config as cms

process = cms.Process('Analysis')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring("rfio:/castor/cern.ch/user/a/antoniov/Summer08/POMWIG_SingleDiffractiveDijetsMinus_10TeV_Pt_30_cff_py_GEN.root")
)

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('PhysicsTools.HepMCCandAlgos.genParticles_cfi')
process.load('RecoJets.Configuration.GenJetParticles_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')

process.genParticles.abortOnUnknownPDGCode = False

process.SDDijets = cms.EDAnalyzer("SDDijetsAnalyzer",
	GenParticleTag = cms.InputTag("genParticles"),
	GenJetTag = cms.InputTag("iterativeCone5GenJets")
)

process.add_(cms.Service("TFileService",
		fileName = cms.string("analysisSDDijetsMinus_Pt30_histos.root")
	)
)

process.analysis = cms.Path(process.genParticles*process.genJetParticles*process.recoGenJets*process.SDDijets)
