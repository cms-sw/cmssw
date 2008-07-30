import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
# globalMuons cand collection
glbCands = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('isGlobalMuon > 0 & pt > 20.0')
)

# generalTracks cand collection
allTracks = cms.EDProducer("TrackViewCandidateProducer",
    src = cms.InputTag("generalTracks"),
    particleType = cms.string('mu+'),
    cut = cms.string('pt > 0')
)

generalCands = cms.EDFilter("RecoChargedCandidateRefSelector",
    src = cms.InputTag("allTracks"),
    cut = cms.string('pt > 10.0')
)

# standAloneMuons cand collection
staTracks = cms.EDProducer("TrackViewCandidateProducer",
    src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    particleType = cms.string('mu+'),
    cut = cms.string('pt > 0')
)

staCands = cms.EDFilter("RecoChargedCandidateRefSelector",
    src = cms.InputTag("staTracks"),
    cut = cms.string('pt > 10.0')
)

#module staCands = MuonRefSelector {
#    InputTag src = muons
#    string cut = "isStandAloneMuon > 0 & pt > 10.0"
#}
# standAloneMuons deltaR matched to generalTracks
staTkMap = cms.EDFilter("TrivialDeltaRViewMatcher",
    src = cms.InputTag("staCands"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("generalCands")
)

staTkMatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag(""),
    ResMatchMapSource = cms.untracked.InputTag("staTkMap"),
    CandidateSource = cms.untracked.InputTag("staCands")
)

# generalTracks deltaR matched to standAloneMuons
tkStaMap = cms.EDFilter("TrivialDeltaRViewMatcher",
    src = cms.InputTag("generalCands"),
    distMin = cms.double(0.3),
    matched = cms.InputTag("staCands")
)

tkStaMatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag(""),
    ResMatchMapSource = cms.untracked.InputTag("tkStaMap"),
    CandidateSource = cms.untracked.InputTag("generalCands")
)

# muon with standAloneMuon inside
glbWithStaTkMatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag("glbCands"),
    ResMatchMapSource = cms.untracked.InputTag(""),
    CandidateSource = cms.untracked.InputTag("staTkMatched")
)

# muon with standAloneMuon inside
glbWithTkMatched = cms.EDFilter("RecoChargedCandidateMatchedProbeMaker",
    Matched = cms.untracked.bool(True),
    ReferenceSource = cms.untracked.InputTag("glbCands"),
    ResMatchMapSource = cms.untracked.InputTag(""),
    CandidateSource = cms.untracked.InputTag("generalCands")
)

# Make the glb_tag tk_probe association map
tagGlbprobeTkMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("glbCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("generalCands")
)

# Make the glb_tag sta_probe association map
tagGlbprobeStaMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("glbCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("staCands")
)

# Make the glb_tag sta_glb association
tagGlbprobeGlbStaMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("glbCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("glbWithStaTkMatched")
)

# Make the glb_tag tk_probe association
tagGlbprobeGlbTkMap = cms.EDProducer("TagProbeProducer",
    MassMaxCut = cms.untracked.double(120.0),
    TagCollection = cms.InputTag("glbCands"),
    MassMinCut = cms.untracked.double(50.0),
    ProbeCollection = cms.InputTag("glbWithTkMatched")
)

# find generator particles matching by DeltaR
glbMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("glbCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

# find generator particles matching by DeltaR
generalMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("generalCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

# find generator particles matching by DeltaR
staMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("staCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

# find generator particles matching by DeltaR
staTkMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("staTkMatched"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

# find generator particles matching by DeltaR
tkStaMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("tkStaMatched"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

# find generator particles matching by DeltaR
glbWithStaMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("glbWithStaTkMatched"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

# find generator particles matching by DeltaR
glbWithTkMCMatch = cms.EDFilter("MCTruthDeltaRMatcherNew",
    pdgId = cms.vint32(13),
    src = cms.InputTag("glbWithTkMatched"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticles")
)

#
#
#
muon_cands = cms.Sequence(genParticles+allTracks+staTracks*glbCands+staCands+generalCands*staTkMap+tkStaMap*staTkMatched+tkStaMatched+glbWithStaTkMatched+glbWithTkMatched*tagGlbprobeTkMap+tagGlbprobeStaMap+tagGlbprobeGlbStaMap+tagGlbprobeGlbTkMap+glbMCMatch+generalMCMatch+staMCMatch+staTkMCMatch+tkStaMCMatch+glbWithStaMCMatch+glbWithTkMCMatch)

