import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
# Make the charged candidate collections from tracks
allTracks = cms.EDProducer("TrackViewCandidateProducer",
    src = cms.InputTag("generalTracks"),
    particleType = cms.string('mu+'),
    cut = cms.string('pt > 0')
)

staTracks = cms.EDProducer("TrackViewCandidateProducer",
    src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    particleType = cms.string('mu+'),
    cut = cms.string('pt > 0')
)

# Make the input candidate collections
tagCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon > 0 & pt > 20.0')
)

# Standalone muon tracks (probes)
staProbCands = cms.EDFilter("RecoChargedCandidateRefSelector",
    src = cms.InputTag("staTracks"),
    cut = cms.string('pt > 10.0')
)

# Tracker muons (to be matched)
trackerCands = cms.EDFilter("RecoChargedCandidateRefSelector",
    src = cms.InputTag("allTracks"),
    cut = cms.string('pt > 10.0')
)

# Match track and stand alone candidates
# to get the passing probe candidates
StaTkMap = cms.EDFilter("TrivialDeltaRViewMatcher",
    src = cms.InputTag("staProbeCands"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("trackerCands")
)

# Use the producer to get a list of matched candidates
StaTkMatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag("trackerCands"),
    ResMatchMapSource = cms.untracked.InputTag("StaTkMap"),
    CandidateSource = cms.untracked.InputTag("staProbeCands")
)

StaTkUnmatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(False),
    ReferenceSource = cms.untracked.InputTag("trackerCands"),
    ResMatchMapSource = cms.untracked.InputTag("StaTkMap"),
    CandidateSource = cms.untracked.InputTag("staProbeCands")
)

# Make the tag probe association map
muonTagProbeMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("tagCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("staProbeCands")
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
    src = cms.InputTag("staProbeCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

passProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("StaTkMatched"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

muon_cands = cms.Sequence(genParticles+allTracks+staTracks*tagCands+trackerCands+cms.SequencePlaceholder("staProbeCands")*StaTkMap*StaTkMatched+StaTkUnmatched+muonTagProbeMap+tagMuonMatch+allProbeMuonMatch+passProbeMuonMatch)

