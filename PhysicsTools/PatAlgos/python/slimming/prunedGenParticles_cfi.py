import FWCore.ParameterSet.Config as cms

prunedGenParticles = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
        "drop  *", # this is the default
        "++keep std::abs(obj.pdgId()) == 11 || std::abs(obj.pdgId()) == 13 || std::abs(obj.pdgId()) == 15", # keep leptons, with history
        "keep std::abs(obj.pdgId()) == 12 || std::abs(obj.pdgId()) == 14 || std::abs(obj.pdgId()) == 16",   # keep neutrinos
        "drop   obj.status() == 2",                                              # drop the shower part of the history
        "+keep obj.pdgId() == 22 && obj.status() == 1 && (obj.pt() > 10 || obj.isPromptFinalState())", # keep gamma above 10 GeV (or all prompt) and its first parent
        "+keep std::abs(obj.pdgId()) == 11 && obj.status() == 1 && (obj.pt() > 3 || obj.isPromptFinalState())", # keep first parent of electrons above 3 GeV (or prompt)
        "keep++ std::abs(obj.pdgId()) == 15",                                         # but keep keep taus with decays
	"drop  obj.status() > 30 && obj.status() < 70 ", 				   #remove pythia8 garbage
	"drop  obj.pdgId() == 21 && obj.pt() < 5",                                    #remove pythia8 garbage
        "drop  obj.status() == 2 && std::abs(obj.pdgId()) == 21",                          # but remove again gluons in the inheritance chain
        "keep std::abs(obj.pdgId()) == 23 || std::abs(obj.pdgId()) == 24 || std::abs(obj.pdgId()) == 25 || std::abs(obj.pdgId()) == 6 || std::abs(obj.pdgId()) == 37 ",   # keep VIP(articles)s
        "keep std::abs(obj.pdgId()) == 310 && std::abs(obj.eta()) < 2.5 && obj.pt() > 1 ",                                                     # keep K0
# keep heavy flavour quarks for parton-based jet flavour
	"keep ( 4 <= std::abs(obj.pdgId()) && std::abs(obj.pdgId()) <= 5 ) && ( obj.status() == 2 || obj.status() == 11 || obj.status() == 71 || obj.status() == 72 )",
# keep light-flavour quarks and gluons for parton-based jet flavour
	"keep ( (1 <= std::abs(obj.pdgId()) && std::abs(obj.pdgId()) <= 3) || obj.pdgId() == 21) && ( obj.status() == 2 || obj.status() == 11 || obj.status() == 71 || obj.status() == 72) && obj.pt()>5", 
# keep b and c hadrons for hadron-based jet flavour
	"keep (400 < std::abs(obj.pdgId()) && std::abs(obj.pdgId()) < 600) || (4000 < std::abs(obj.pdgId()) && std::abs(obj.pdgId()) < 6000)",
# additional c hadrons for jet fragmentation studies
	"keep std::abs(obj.pdgId()) == 10411 || std::abs(obj.pdgId()) == 10421 || std::abs(obj.pdgId()) == 10413 || std::abs(obj.pdgId()) == 10423 || std::abs(obj.pdgId()) == 20413 || std::abs(obj.pdgId()) == 20423 || std::abs(obj.pdgId()) == 10431 || std::abs(obj.pdgId()) == 10433 || std::abs(obj.pdgId()) == 20433", 
# additional b hadrons for jet fragmentation studies
	"keep std::abs(obj.pdgId()) == 10511 || std::abs(obj.pdgId()) == 10521 || std::abs(obj.pdgId()) == 10513 || std::abs(obj.pdgId()) == 10523 || std::abs(obj.pdgId()) == 20513 || std::abs(obj.pdgId()) == 20523 || std::abs(obj.pdgId()) == 10531 || std::abs(obj.pdgId()) == 10533 || std::abs(obj.pdgId()) == 20533 || std::abs(obj.pdgId()) == 10541 || std::abs(obj.pdgId()) == 10543 || std::abs(obj.pdgId()) == 20543", 
#keep SUSY particles
	"keep ( 1000001 <= std::abs(obj.pdgId()) && std::abs(obj.pdgId()) <= 1000039 ) || ( 2000001 <= std::abs(obj.pdgId()) && std::abs(obj.pdgId()) <= 2000015)",
# keep protons 
        "keep obj.pdgId() == 2212",
        "keep obj.status() == 3 || ( 21 <= obj.status() && obj.status() <= 29 ) || ( 11 <= obj.status() && obj.status() <= 19 )",  #keep event summary (status=3 for pythia6, 21 <= obj.status() && obj.status() <= 29 for pythia8)
        "keep obj.isHardProcess() || obj.fromHardProcessFinalState() || obj.fromHardProcessDecayed() || obj.fromHardProcessBeforeFSR() || (obj.statusFlags().fromHardProcess() && obj.statusFlags().isLastCopy())",  #keep event summary based on obj.status() flags
    )
)
