import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
)

process.caloJetCollectionClone = cms.EDProducer("CaloJetShallowCloneProducer",
    src = cms.InputTag("iterativeCone5CaloJets")
)

process.genJetCollectionClone = cms.EDProducer("GenJetShallowCloneProducer",
    src = cms.InputTag("iterativeCone5GenJets")
)

process.caloJetSele = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("caloJetCollectionClone"),
    ptMin = cms.double(20.0)
)

process.genJetSele = cms.EDFilter("PtMinCandSelector",
    src = cms.InputTag("genJetCollectionClone"),
    ptMin = cms.double(20.0)
)

process.jetMatchOne = cms.EDFilter("CandOneToOneDeltaRMatcher",
    src = cms.InputTag("iterativeCone5GenJets"),
    algoMethod = cms.string('SwitchMode'),
    matched = cms.InputTag("iterativeCone5CaloJets")
)

process.jetMatchMany = cms.EDFilter("CandOneToManyDeltaRMatcher",
    printDebug = cms.untracked.bool(True),
    src = cms.InputTag("genJetSele"),
    matched = cms.InputTag("caloJetSele")
)

process.printJet = cms.EDFilter("jetMatch",
    src = cms.InputTag("genJetSele"),
    matchMapMany = cms.InputTag("jetMatchMany"),
    HistOutFile = cms.untracked.string('myPlots.root'),
    matchMapOne = cms.InputTag("jetMatchOne","src2mtc"),
    matched = cms.InputTag("caloJetSele")
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.caloJetCollectionClone*process.genJetCollectionClone*process.caloJetSele*process.genJetSele*process.jetMatchOne*process.jetMatchMany*process.printJet)
process.outpath = cms.EndPath(process.printEventNumber)
process.MessageLogger.cerr.default.limit = 10


