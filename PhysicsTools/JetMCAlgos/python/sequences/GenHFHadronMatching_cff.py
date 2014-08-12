import FWCore.ParameterSet.Config as cms


from RecoJets.Configuration.RecoGenJets_cff import ak5GenJets
from PhysicsTools.JetMCAlgos.GenJetParticles_cfi import genParticlesForJetsPlusNoHadron
from PhysicsTools.JetMCAlgos.GenJetParticles_cff import genParticlesForJetsNoNuPlusNoHadron
from PhysicsTools.JetMCAlgos.GenHFHadronMatcher_cfi import matchGenHFHadron


# Supplies PDG ID to real name resolution of MC particles
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *




# Configuration for matching B-hadrons ================================================================
genParticlesForJetsPlusBHadron = genParticlesForJetsPlusNoHadron.clone()
genParticlesForJetsPlusBHadron.injectHadronFlavours = [5]
ak5GenJetsPlusBHadron = ak5GenJets.clone()
ak5GenJetsPlusBHadron.src = "genParticlesForJetsPlusBHadron"
matchGenBHadron = matchGenHFHadron.clone()
matchGenBHadron.flavour = 5
matchGenBHadron.genJets = cms.InputTag("ak5GenJetsPlusBHadron", "", "")
genBHadronMatchingSequence = cms.Sequence( genParticlesForJetsPlusBHadron * ak5GenJetsPlusBHadron * matchGenBHadron )


# Configuration for matching C-hadrons =================================================================
genParticlesForJetsPlusCHadron = genParticlesForJetsPlusNoHadron.clone()
genParticlesForJetsPlusCHadron.injectHadronFlavours = [4]
ak5GenJetsPlusCHadron = ak5GenJets.clone()
ak5GenJetsPlusCHadron.src = "genParticlesForJetsPlusCHadron"
matchGenCHadron = matchGenHFHadron.clone()
matchGenCHadron.flavour = 4
matchGenCHadron.genJets = cms.InputTag("ak5GenJetsPlusCHadron", "", "")
genCHadronMatchingSequence = cms.Sequence( genParticlesForJetsPlusCHadron * ak5GenJetsPlusCHadron * matchGenCHadron )


# Configuration for matching B- and C-hadrons ==========================================================
genParticlesForJetsPlusBCHadron = genParticlesForJetsPlusNoHadron.clone()
genParticlesForJetsPlusBCHadron.injectHadronFlavours = [5, 4]
ak5GenJetsPlusBCHadron = ak5GenJets.clone()
ak5GenJetsPlusBCHadron.src = "genParticlesForJetsPlusBCHadron"
matchGenBCHadronB = matchGenBHadron.clone()
matchGenBCHadronB.genJets = cms.InputTag("ak5GenJetsPlusBCHadron", "", "")
matchGenBCHadronC = matchGenCHadron.clone()
matchGenBCHadronC.genJets = cms.InputTag("ak5GenJetsPlusBCHadron", "", "")
genBCHadronMatchingSequence = cms.Sequence( genParticlesForJetsPlusBCHadron * ak5GenJetsPlusBCHadron * matchGenBCHadronB * matchGenBCHadronC )


# Configuration for matching B-hadrons (jets without neutrinos) ========================================
genParticlesForJetsNoNuPlusBHadron = genParticlesForJetsNoNuPlusNoHadron.clone()
genParticlesForJetsNoNuPlusBHadron.injectHadronFlavours = [5]
ak5GenJetsNoNuPlusBHadron = ak5GenJets.clone()
ak5GenJetsNoNuPlusBHadron.src = "genParticlesForJetsNoNuPlusBHadron"
matchGenBHadronNoNu = matchGenBHadron.clone()
matchGenBHadronNoNu.genJets = cms.InputTag("ak5GenJetsNoNuPlusBHadron", "", "")
genBHadronMatchingNoNuSequence = cms.Sequence( genParticlesForJetsNoNuPlusBHadron * ak5GenJetsNoNuPlusBHadron * matchGenBHadronNoNu )


# Configuration for matching C-hadrons (jets without neutrinos) ========================================
genParticlesForJetsNoNuPlusCHadron = genParticlesForJetsNoNuPlusNoHadron.clone()
genParticlesForJetsNoNuPlusCHadron.injectHadronFlavours = [4]
ak5GenJetsNoNuPlusCHadron = ak5GenJets.clone()
ak5GenJetsNoNuPlusCHadron.src = "genParticlesForJetsNoNuPlusCHadron"
matchGenCHadronNoNu = matchGenCHadron.clone()
matchGenCHadronNoNu.genJets = cms.InputTag("ak5GenJetsNoNuPlusCHadron", "", "")
genCHadronMatchingNoNuSequence = cms.Sequence( genParticlesForJetsNoNuPlusCHadron * ak5GenJetsNoNuPlusCHadron * matchGenCHadronNoNu )


# Configuration for matching B- and C-hadrons (jets without neutrinos) =================================
genParticlesForJetsNoNuPlusBCHadron = genParticlesForJetsNoNuPlusNoHadron.clone()
genParticlesForJetsNoNuPlusBCHadron.injectHadronFlavours = [5, 4]
ak5GenJetsNoNuPlusBCHadron = ak5GenJets.clone()
ak5GenJetsNoNuPlusBCHadron.src = "genParticlesForJetsNoNuPlusBCHadron"
matchGenBCHadronBNoNu = matchGenBHadron.clone()
matchGenBCHadronBNoNu.genJets = cms.InputTag("ak5GenJetsNoNuPlusBCHadron", "", "")
matchGenBCHadronCNoNu = matchGenCHadron.clone()
matchGenBCHadronCNoNu.genJets = cms.InputTag("ak5GenJetsNoNuPlusBCHadron", "", "")
genBCHadronMatchingNoNuSequence = cms.Sequence( genParticlesForJetsNoNuPlusBCHadron * ak5GenJetsNoNuPlusBCHadron * matchGenBCHadronBNoNu * matchGenBCHadronCNoNu )


