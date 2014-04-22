import FWCore.ParameterSet.Config as cms

prunedGenParticles = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
        "drop  *", # this is the default
        "keep status == 3",  #keep event summary status3 (for pythia)
        "++keep abs(pdgId) == 11 || abs(pdgId) == 13 || abs(pdgId) == 15", # keep leptons, with history
        "keep abs(pdgId) == 12 || abs(pdgId) == 14 || abs(pdgId) == 16", # keep neutrinos
        "++keep pdgId == 22 && status == 1 && pt > 10",                    # keep gamma above 10 GeV
        "drop   status == 2",                                              # drop the shower part of the history
        "keep++ abs(pdgId) == 15",                                         # but keep keep taus with decays
        "++keep  4 <= abs(pdgId) <= 6 ",                                   # keep also heavy quarks
        "++keep  (400 < abs(pdgId) < 600) || (4000 < abs(pdgId) < 6000)",  # and their hadrons
        "drop   status == 2 && abs(pdgId) == 21",                          # but remove again gluons in the inheritance chain
        "keep  (1 <= abs(pdgId) <= 3 || abs(pdgId) == 21) && pt > 10",     # keep hard partons
        "keep abs(pdgId) == 23 || abs(pdgId) == 24 || abs(pdgId) == 25  || abs(pdgId) == 37 ",   # keep V.I.P.s
    )
)
