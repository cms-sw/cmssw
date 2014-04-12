import FWCore.ParameterSet.Config as cms

process = cms.Process("TagProbe")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.options   = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
)
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource", 
    fileNames = cms.untracked.vstring('file:ZMM_7TeV_cfi_py_GEN.root'),
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
)    

# Convert genParticles to PATobjects
process.load("PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi")
process.patGenericParticles.src = cms.InputTag("genParticles")

# Tag collection
process.tags = cms.EDProducer("PATGenericParticleRefSelector",
    src = cms.InputTag("patGenericParticles"),
    cut = cms.string("abs(pdgId) = 13 & status = 1 & phi>0")
)

# Probe collection
process.probes = cms.EDProducer("PATGenericParticleRefSelector",
    src = cms.InputTag("patGenericParticles"),
    cut = cms.string("status = 1"), # make some background
)

# Tag and Probe pairs
process.tagProbePairs = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("tags@+ probes@-"), # charge coniugate states are implied
    cut   = cms.string("40 < mass"),
)

# Make the fit tree and save it in the "MuonID" directory
process.MuonID = cms.EDAnalyzer("TagProbeFitTreeProducer",
    tagProbePairs = cms.InputTag("tagProbePairs"),
    arbitration   = cms.string("None"),
    variables = cms.PSet(
        pt  = cms.string("pt"),
        eta = cms.string("eta"),
    ),
    flags = cms.PSet(
        muon = cms.string("((abs(pdgId) = 13) & phi>0) | (abs(pdgId) != 13 & phi>2)"),
    ),
    isMC = cms.bool(False),
    tagMatches = cms.InputTag("muMcMatch"),
    probeMatches  = cms.InputTag("muMcMatch"),
    motherPdgId = cms.int32(443),
    makeMCUnbiasTree = cms.bool(False),
    #checkMotherInUnbiasEff = cms.bool(True),
    #allProbes     = cms.InputTag("trkProbes"),
    addRunLumiInfo = cms.bool(True),
)

process.tagAndProbe = cms.Path( 
    process.patGenericParticles * 
    process.tags *
    process.probes *
    process.tagProbePairs * 
    process.MuonID
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("testTagProbeFitTreeProducer_GEN.root"))
