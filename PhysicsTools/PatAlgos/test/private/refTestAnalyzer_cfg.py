# The following comments couldn't be translated into the new config version:

# discard jets that match with clean electrons
import FWCore.ParameterSet.Config as cms

process = cms.Process("TestTag2Map")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:in.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.cleanElectrons = cms.EDFilter("PATElectronCleaner",
    removeDuplicates = cms.bool(True),
    electronSource = cms.InputTag("pixelMatchGsfElectrons")
)

process.cleanJets = cms.EDFilter("PATCaloJetCleaner",
    removeOverlaps = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        collection = cms.InputTag("cleanElectrons")
    )),
    jetSource = cms.InputTag("iterativeCone5CaloJets")
)

process.convBtag = cms.EDFilter("JetTagToValueMapFloat",
    src = cms.InputTag("iterativeCone5CaloJets"),
    tags = cms.InputTag("jetProbabilityJetTags")
)

process.skimBtag = cms.EDFilter("CandValueMapSkimmerFloat",
    association = cms.InputTag("convBtag","jetProbabilityJetTags"),
    collection = cms.InputTag("cleanJets"),
    backrefs = cms.InputTag("cleanJets")
)

process.refTestAnalyzer = cms.EDAnalyzer("RefTestAnalyzer",
    jets1 = cms.InputTag("cleanJets"),
    jets0 = cms.InputTag("iterativeCone5CaloJets"),
    btag = cms.InputTag("convBtag","jetProbabilityJetTags"),
    newbtag = cms.InputTag("skimBtag"),
    backmap = cms.InputTag("cleanJets"),
    jtag = cms.InputTag("jetProbabilityJetTags")
)

process.p = cms.Path(process.cleanElectrons*process.cleanJets*process.convBtag*process.skimBtag*process.refTestAnalyzer)


