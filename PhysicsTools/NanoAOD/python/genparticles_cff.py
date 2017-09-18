import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *



##################### User floats producers, selectors ##########################

finalGenParticles = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("prunedGenParticles"),
    select = cms.vstring(
        "drop  *", # this is the default
        "++keep abs(pdgId) == 11 || abs(pdgId) == 13 || abs(pdgId) == 15", # keep leptons, with history
        "drop   status == 2",                                              # drop the shower part of the history
        "keep (400 < abs(pdgId) < 600) || (4000 < abs(pdgId) < 6000)", 
        "keep abs(pdgId) == 11 || abs(pdgId) == 13 || abs(pdgId) == 15",   # keep leptons (also status1)
        "keep abs(pdgId) == 12 || abs(pdgId) == 14 || abs(pdgId) == 16",   # keep neutrinos
        "+keep pdgId == 22 && status == 1 && (pt > 10 || isPromptFinalState())", # keep gamma above 10 GeV (or all prompt) and its first parent
        "+keep abs(pdgId) == 11 && status == 1 && (pt > 3 || isPromptFinalState())", # keep first parent of electrons above 3 GeV (or prompt)
        "keep++ abs(pdgId) == 15",                                         # but keep keep taus with decays
        "drop  status > 30 && status < 70 ",                               # remove pythia8 garbage
        "drop  pdgId == 21 && pt < 5",                                     # remove pythia8 garbage
        "drop   status == 2 && abs(pdgId) == 21",                          # but remove again gluons in the inheritance chain
        "keep abs(pdgId) == 23 || abs(pdgId) == 24 || abs(pdgId) == 25 || abs(pdgId) == 6 || abs(pdgId) == 37 ",   # keep VIP(articles)s
        "keep abs(pdgId) == 310 && abs(eta) < 2.5 && pt > 1 ",                                                     # keep K0
        "+keep abs(pdgId) == 13 && status == 1", # keep muon parents
        "keep (4 <= abs(pdgId) <= 5) && ( status <30 || status > 70)",
        "keep (1 <= abs(pdgId) <= 3 || pdgId = 21) & (status = 2 || status = 11 || status = 71 || status = 72) && pt>15", #increased to 15GeV wrt miniAOD
# keep onia states, phi, X(3872), Z(4430)+ and psi(4040)
#        "keep+ abs(pdgId) == 333",
#        "keep+ abs(pdgId) == 9920443 || abs(pdgId) == 9042413 || abs(pdgId) == 9000443",
#        "keep+ abs(pdgId) == 443 || abs(pdgId) == 100443 || abs(pdgId) == 10441 || abs(pdgId) == 20443 || abs(pdgId) == 445 || abs(pdgId) == 30443",
#        "keep+ abs(pdgId) == 553 || abs(pdgId) == 100553 || abs(pdgId) == 200553 || abs(pdgId) == 10551 || abs(pdgId) == 20553 || abs(pdgId) == 555",
# additional c hadrons for jet fragmentation studies
#        "keep abs(pdgId) = 10411 || abs(pdgId) = 10421 || abs(pdgId) = 10413 || abs(pdgId) = 10423 || abs(pdgId) = 20413 || abs(pdgId) = 20423 || abs(pdgId) = 10431 || abs(pdgId) = 10433 || abs(pdgId) = 20433",
# additional b hadrons for jet fragmentation studies
#        "keep abs(pdgId) = 10511 || abs(pdgId) = 10521 || abs(pdgId) = 10513 || abs(pdgId) = 10523 || abs(pdgId) = 20513 || abs(pdgId) = 20523 || abs(pdgId) = 10531 || abs(pdgId) = 10533 || abs(pdgId) = 20533 || abs(pdgId) = 10541 || abs(pdgId) = 10543 || abs(pdgId) = 20543",
#keep SUSY particles
        "keep (1000001 <= abs(pdgId) <= 1000039 ) || ( 2000001 <= abs(pdgId) <= 2000015)",
## keep protons 
#        "keep pdgId = 2212",
        "keep status == 3 || ( 21 <= status <= 29) || ( 11 <= status <= 19)",  #keep event summary (status=3 for pythia6, 21 <= status <= 29 for pythia8)
        "keep isHardProcess() || fromHardProcessFinalState() || fromHardProcessDecayed() || fromHardProcessBeforeFSR() || (statusFlags().fromHardProcess() && statusFlags().isLastCopy())",  #keep event summary based on status flags
    )
)



##################### Tables for final output and docs ##########################
genParticleTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("finalGenParticles"),
    cut = cms.string(""), #we should not filter after pruning
    name= cms.string("GenPart"),
    doc = cms.string("interesting gen particles "),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the taus
    variables = cms.PSet(
         pt  = Var("pt",  float,precision=8),
         phi = Var("phi", float,precision=8),
         eta  = Var("eta",  float,precision=8),
	 pdgId  = Var("pdgId", int, doc="PDG id"),
	 status  = Var("status", int, doc="Particle status. 1=stable"),
	 genPartIdxMother = Var("?numberOfMothers>0?motherRef(0).key():-1", int, doc="index of the mother particle"),
 
    )
)

genParticleSequence = cms.Sequence(finalGenParticles)
genParticleTables = cms.Sequence(genParticleTable)

