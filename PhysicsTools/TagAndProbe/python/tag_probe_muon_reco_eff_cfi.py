import FWCore.ParameterSet.Config as cms

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
staCands = cms.EDFilter("RecoChargedCandidateRefSelector",
    src = cms.InputTag("staTracks"),
    cut = cms.string('pt > 10.0')
)

# Tracker muons (to be matched)
tkProbeCands = cms.EDFilter("RecoChargedCandidateRefSelector",
    src = cms.InputTag("allTracks"),
    cut = cms.string('pt > 10.0')
)

# Match track and stand alone candidates
# to get the passing probe candidates
TkStaMap = cms.EDFilter("TrivialDeltaRViewMatcher",
    src = cms.InputTag("tkProbeCands"),
    distMin = cms.double(0.15),
    matched = cms.InputTag("staCands")
)

# Use the producer to get a list of matched candidates
TkStaMatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag("staCands"),
    ResMatchMapSource = cms.untracked.InputTag("TkStaMap"),
    CandidateSource = cms.untracked.InputTag("tkProbeCands")
)

TkStaUnmatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(False),
    ReferenceSource = cms.untracked.InputTag("staCands"),
    ResMatchMapSource = cms.untracked.InputTag("TkStaMap"),
    CandidateSource = cms.untracked.InputTag("tkProbeCands")
)

# Make the tag probe association map
muonTagProbeMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("tagCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("tkProbeCands"),
    PassingProbeCollection = cms.InputTag("TkStaMatched")
)

# find generator particles matching by DeltaR
tagMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("tagCands"),
    distMin = cms.double(0.15),
    matched = cms.InputTag("genParticles")
)

allProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("tkProbeCands"),
    distMin = cms.double(0.15),
    matched = cms.InputTag("genParticles")
)

passProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("TkStaMatched"),
    distMin = cms.double(0.15),
    matched = cms.InputTag("genParticles")
)

muon_cands = cms.Sequence(allTracks+staTracks*tagCands+tkProbeCands+staCands*TkStaMap*TkStaMatched+TkStaUnmatched+muonTagProbeMap+tagMuonMatch+allProbeMuonMatch+passProbeMuonMatch)
