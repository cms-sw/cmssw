import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
# Make the input tag candidate collections
tagCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon > 0 & pt > 20.0')
)

# Make the input GM Prob candidate collections
gmProbCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon > 0')
)

# Make the input GM Iso Prob candidate collections
gmIsoCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon > 0 & isIsolationValid>0')
)

# Match gm track and gm Iso  candidates
# to get the passing probe candidates
gmgmIsoMap = cms.EDFilter("TrivialDeltaRViewMatcher",
    src = cms.InputTag("gmProbeCands"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("gmIsoCands")
)

# Use the producer to get a list of matched candidates
gmgmIsoMatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag("gmIsoCands"),
    ResMatchMapSource = cms.untracked.InputTag("gmgmIsoMap"),
    CandidateSource = cms.untracked.InputTag("gmProbeCands")
)

gmgmIsoUnmatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(False),
    ReferenceSource = cms.untracked.InputTag("gmIsoCands"),
    ResMatchMapSource = cms.untracked.InputTag("gmgmIsoMap"),
    CandidateSource = cms.untracked.InputTag("gmProbeCands")
)

# Make the tag probe association map
muonTagProbeMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("tagCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("gmProbeCands")
)

# find generator particles matching by DeltaR
tagMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("tagCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

allProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("gmProbeCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

passProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("gmgmIsoMatched"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

muon_cands = cms.Sequence(cms.SequencePlaceholder("genParticlies")+tagCands+cms.SequencePlaceholder("gmProbeCands")+gmIsoCands*gmgmIsoMap*gmgmIsoMatched+gmgmIsoUnmatched+muonTagProbeMap+tagMuonMatch+allProbeMuonMatch+passProbeMuonMatch)

