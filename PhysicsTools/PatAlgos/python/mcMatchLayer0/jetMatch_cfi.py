import FWCore.ParameterSet.Config as cms

#
# Example for a configuration of the MC match
#
patJetPartonMatch = cms.EDProducer("MCMatcher",      # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = cms.InputTag("ak4PFJetsCHS"),      # RECO objects to match
    matched     = cms.InputTag("genParticles"),      # mc-truth particle collection
    mcPdgId     = cms.vint32(1, 2, 3, 4, 5, 21),     # one or more PDG ID (quarks except top; gluons)
    mcStatus    = cms.vint32(3,23),                  # PYTHIA6/Herwig++ status code (3 = hard scattering), 23 in Pythia8
    checkCharge = cms.bool(False),                   # False = any value of the charge of MC and RECO is ok
    maxDeltaR   = cms.double(0.4),                   # Minimum deltaR for the match
    maxDPtRel   = cms.double(3.0),                   # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),          # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),         # False = just match input in order; True = pick lowest deltaR pair first
)

patJetGenJetMatch = cms.EDProducer("GenJetMatcher",  # cut on deltaR; pick best by deltaR
    src         = cms.InputTag("ak4PFJetsCHS"),      # RECO jets (any View<Jet> is ok)
    matched     = cms.InputTag("ak4GenJets"),        # GEN jets  (must be GenJetCollection)
    mcPdgId     = cms.vint32(),                      # n/a
    mcStatus    = cms.vint32(),                      # n/a
    checkCharge = cms.bool(False),                   # n/a
    maxDeltaR   = cms.double(0.4),                   # Minimum deltaR for the match
    #maxDPtRel   = cms.double(3.0),                  # Minimum deltaPt/Pt for the match (not used in GenJetMatcher)
    resolveAmbiguities    = cms.bool(True),          # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(False),         # False = just match input in order; True = pick lowest deltaR pair first
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(patJetGenJetMatch,
                                           maxDeltaR = 0.4,
                                           resolveByMatchQuality = True,
                                           src = "akCs4PFJets",
                                       )

(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(patJetPartonMatch, src = "akCs4PFJets")
