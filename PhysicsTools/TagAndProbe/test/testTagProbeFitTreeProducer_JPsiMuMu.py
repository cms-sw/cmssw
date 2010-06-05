import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.options   = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring('file:JpsiMM_7TeV_cfi_py_GEN_FASTSIM.root'),
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
)    

# Merge calomuons into the collection of muons
from RecoMuon.MuonIdentification.calomuons_cfi import calomuons;
process.allMuons = cms.EDProducer("CaloMuonMerger",
    muons     = cms.InputTag("muons"), 
    caloMuons = cms.InputTag("calomuons"),
    minCaloCompatibility = calomuons.minCaloCompatibility,
)

# Match to MC truth
process.muMcMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("allMuons"),
    distMin = cms.double(0.1),
    matched = cms.InputTag("genParticles"),
)

# Tag collection
process.tagMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("allMuons"),
    cut = cms.string("isGlobalMuon"), 
)

# Probe collection
process.caloProbes = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("allMuons"),
    cut = cms.string("isCaloMuon"), 
)

# Passing Probe collection
process.caloPassingGlb = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("allMuons"),
    cut = cms.string("isCaloMuon && isGlobalMuon"), 
)

# Tag and Probe pairs
process.tagProbePairs = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("tagMuons@+ caloProbes@-"), # charge coniugate states are implied
    cut   = cms.string("2.6 < mass < 3.6"),
)

# Make the fit tree and save it in the "MuonID" directory
process.MuonID = cms.EDAnalyzer("TagProbeFitTreeProducer",
    tagProbePairs = cms.InputTag("tagProbePairs"),
    arbitration   = cms.string("OneProbe"),
    variables = cms.PSet(
        pt  = cms.string("pt"),
        eta = cms.string("eta"),
        abseta = cms.string("abs(eta)"),
    ),
    flags = cms.PSet(
        Glb = cms.InputTag("caloPassingGlb"),
        TM = cms.string("isTrackerMuon"),
    ),
    isMC = cms.bool(True),
    tagMatches = cms.InputTag("muMcMatch"),
    probeMatches  = cms.InputTag("muMcMatch"),
    motherPdgId = cms.int32(443),
    makeMCUnbiasTree = cms.bool(False),
    #checkMotherInUnbiasEff = cms.bool(True),
    #allProbes     = cms.InputTag("trkProbes"),
    addRunLumiInfo = cms.bool(True),
)

process.tagAndProbe = cms.Path( 
    process.allMuons *
    process.muMcMatch *
    process.tagMuons *
    process.caloProbes *
    process.caloPassingGlb *
    process.tagProbePairs * 
    process.MuonID
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("testTagProbeFitTreeProducer_JPsiMuMu.root"))
