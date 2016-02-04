import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/08765709-5826-DD11-9CE8-000423D94700.root')
)

process.genParticlesForJets = cms.EDProducer("InputGenJetsParticleSelector",
    src = cms.InputTag("genParticles"),
    ignoreParticleIDs = cms.vuint32(1000022, 2000012, 2000014, 2000016, 1000039, 
        5000039, 4000012, 9900012, 9900014, 9900016, 
        39),
    partonicFinalState = cms.bool(True),
    excludeResonances = cms.bool(True),
    excludeFromResonancePids = cms.vuint32(12, 13, 14, 16),
    tausAsJets = cms.bool(False)
)

process.partonJet = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("genParticlesForJets"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.0),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5PartonGenJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)

process.selectedJets = cms.EDFilter("GenJetRefSelector",
    src = cms.InputTag("partonJet"),
    cut = cms.string('pt > 20')
)

process.printEvent = cms.EDAnalyzer("printPartonJet",
    src = cms.InputTag("selectedJets"),
    HistOutFile = cms.untracked.string('myPlots.root')
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.genParticlesForJets*process.partonJet*process.selectedJets*process.printEvent)
process.outpath = cms.EndPath(process.printEventNumber)
process.MessageLogger.cerr.default.limit = 1


