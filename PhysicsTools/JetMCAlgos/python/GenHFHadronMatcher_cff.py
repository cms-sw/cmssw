import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cfi import matchGenHFHadron


# Configuration for matching B-hadrons ================================================================
matchGenBHadron = matchGenHFHadron.clone(
    flavour = 5
)

# Configuration for matching C-hadrons =================================================================
matchGenCHadron = matchGenHFHadron.clone(
    flavour = 4
)

