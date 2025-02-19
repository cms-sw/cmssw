import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
)

process.printTree1 = cms.EDFilter("ParticleTreeDrawer",
    status = cms.untracked.vint32(3),
    src = cms.InputTag("genParticles"),
    printP4 = cms.untracked.bool(False),
    printStatus = cms.untracked.bool(False),
    printIndex = cms.untracked.bool(False),
    printVertex = cms.untracked.bool(False),
    printPtEtaPhi = cms.untracked.bool(True)
)

process.printTree2 = cms.EDFilter("ParticleListDrawer",
    src = cms.InputTag("genParticles"),
    maxEventsToPrint = cms.untracked.int32(1)
)

process.genJetSele = cms.EDFilter("GenJetRefSelector",
    src = cms.InputTag("iterativeCone5GenJets"),
    cut = cms.string('pt > 20.0')
)

process.printEvent = cms.EDFilter("printEvent",
    src = cms.InputTag("genJetSele")
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.printTree1*process.printTree2*process.genJetSele*process.printEvent)
process.outpath = cms.EndPath(process.printEventNumber)
process.MessageLogger.cerr.default.limit = 10


