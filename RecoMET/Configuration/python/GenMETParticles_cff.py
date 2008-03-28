import FWCore.ParameterSet.Config as cms

# File: GenMET.cff
# Author: R. Cavanaugh
# Date: 08.08.2006
#
# Form Missing ET from Generator Information and store into event as a GenMET
# product.  Exclude calo invisible final state particles like neutrinos, muons
# To be resolved:  How to exclude exotics, like the LSP?
#
# F.R. Mar. 22, 2007 IMPORTANT: this configuration assumes that some
#                    GenParticle collections are made via GenJet's configuration
from PhysicsTools.HepMCCandAlgos.genParticleCandidatesFast_cfi import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
#  module genCandidatesForMET = GenParticleCandidateSelector
#  {
#      string src = "genParticleCandidates"
#      bool stableOnly = true
#      untracked bool verbose = true
#      vstring excludeList = {"nu_e", "nu_mu", "nu_tau", "mu-",
#                          "~chi_10", 
#                          "~nu_eR", "~nu_muR", "~nu_tauR", 
#			  "Graviton", "~Gravitino", 
#                          "nu_Re", "nu_Rmu", "nu_Rtau", 
#                          "nu*_e0", "Graviton*"
#                         }
#      vstring includeList = {}
#  }
genCandidatesForMET = cms.EDFilter("GenJetParticleRefSelector",
    includeList = cms.vstring(),
    src = cms.InputTag("genParticles"),
    stableOnly = cms.bool(True),
    verbose = cms.untracked.bool(True),
    excludeList = cms.vstring('nu_e', 'nu_mu', 'nu_tau', 'mu-', '~chi_10', '~nu_eR', '~nu_muR', '~nu_tauR', 'Graviton', '~Gravitino', 'nu_Re', 'nu_Rmu', 'nu_Rtau', 'nu*_e0', 'Graviton*')
)

genMETParticles = cms.Sequence(genCandidatesForMET)

