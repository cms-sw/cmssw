import FWCore.ParameterSet.Config as cms

from MuonAnalysis.MuonAssociators.muonL1Match_cfi import *

muonL1MultiMatch = cms.EDProducer("L1MuonMultiMatcher",
    # main configurables, in to set for the propagation and the default match
    muonL1MatcherParameters,

    otherMatchers = cms.PSet(
        ByQ  = muonL1MatcherParameters.clone(sortBy = "quality"),
    ),

    # Reconstructed muons
    src = cms.InputTag("muons"),

    # L1 Muon collection, and preselection on that collection
    matched      = cms.InputTag("l1extraParticles"),
    preselection = cms.string("bx == 0"),
)
