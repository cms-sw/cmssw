import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
# Make the input muon candidate collections
tagCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon > 0 & pt > 20.0')
)

# Standalone muons
staCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isStandAloneMuon > 0 & pt > 10.0')
)

# Tracker muons (probes)
probeCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isTrackerMuon > 0 & pt > 10.0')
)

# Match track and stand alone candidates
# to get the passing probe candidates
tkStaMap = cms.EDFilter("TrivialDeltaRViewMatcher",
    src = cms.InputTag("probeCands"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("staCands")
)

# Use the producer to get a list of matched candidates
# Passing probe candidates ...
tkStaMatched = cms.EDFilter("MuonMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag("staCands"),
    ResMatchMapSource = cms.untracked.InputTag("tkStaMap"),
    CandidateSource = cms.untracked.InputTag("probeCands")
)

# Not used ... only to illustrate how to get unmatched 
tkStaUnmatched = cms.EDFilter("MuonMatchedProbeMaker",
    Matched = cms.untracked.bool(False),
    ReferenceSource = cms.untracked.InputTag("staCands"),
    ResMatchMapSource = cms.untracked.InputTag("tkStaMap"),
    CandidateSource = cms.untracked.InputTag("probeCands")
)

# Make the tag probe association map
muonTagProbeMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("tagCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("probeCands")
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
    src = cms.InputTag("probeCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

passProbeMuonMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("tkStaMatched"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

muon_cands = cms.Sequence(genParticles+tagCands+staCands+probeCands*tkStaMap*tkStaMatched+tkStaUnmatched+muonTagProbeMap+tagMuonMatch+allProbeMuonMatch+passProbeMuonMatch)

