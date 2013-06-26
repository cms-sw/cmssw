import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
)

process.caloJetSele = cms.EDFilter("CaloJetRefSelector",
    src = cms.InputTag("iterativeCone5CaloJets"),
    cut = cms.string('pt > 20.0')
)

process.genJetSele = cms.EDFilter("GenJetRefSelector",
    src = cms.InputTag("iterativeCone5GenJets"),
    cut = cms.string('pt > 20.0')
)

process.jetMatchOne = cms.EDFilter("CandOneToOneDeltaRMatcher",
    src = cms.InputTag("genJetSele"),
    algoMethod = cms.string('SwitchMode'),
    matched = cms.InputTag("caloJetSele")
)

process.printJet = cms.EDFilter("matchOneToOne",
    src = cms.InputTag("genJetSele"),
    matchMapOne1 = cms.InputTag("jetMatchOne","src2mtc"),
    HistOutFile = cms.untracked.string('myPlots1.root'),
    matchMapOne2 = cms.InputTag("jetMatchOne","mtc2src"),
    matched = cms.InputTag("caloJetSele")
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.caloJetSele*process.genJetSele*process.jetMatchOne*process.printJet)
process.outpath = cms.EndPath(process.printEventNumber)


