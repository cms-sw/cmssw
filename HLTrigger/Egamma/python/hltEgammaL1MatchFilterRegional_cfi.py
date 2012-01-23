import FWCore.ParameterSet.Config as cms

hltEgammaL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("l1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltRecoIsolatedEcalCandidate"),
    region_phi_size = cms.double(1.044),
    ncandcut = cms.int32(1),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("theL1SeedFilter"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("l1extraParticles","NonIsolated"),
    saveTags = cms.bool( False )
)


