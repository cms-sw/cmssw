import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer


##################### User floats producers, selectors ##########################

finalGenParticles = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("prunedGenParticles"),
    select = cms.vstring(
        "drop  *", # this is the default
        "++keep abs(pdgId) == 11 || abs(pdgId) == 13 || abs(pdgId) == 15", # keep leptons, with history
        "keep abs(pdgId) == 12 || abs(pdgId) == 14 || abs(pdgId) == 16",   # keep neutrinos
        "drop   status == 2",                                              # drop the shower part of the history
        "+keep pdgId == 22 && status == 1 && pt > 10",                     # keep gamma above 10 GeV and its first parent
        "+keep pdgId == 11 && status == 1 && pt > 3",                     # keep first parent of electrons above 10 GeV
        "keep++ abs(pdgId) == 15",                                         # but keep keep taus with decays
    "drop  status > 30 && status < 70 ",                   #remove pythia8 garbage
    "drop  pdgId == 21 && pt < 5",                                    #remove pythia8 garbage
        "drop   status == 2 && abs(pdgId) == 21",                          # but remove again gluons in the inheritance chain
        "keep abs(pdgId) == 23 || abs(pdgId) == 24 || abs(pdgId) == 25 || abs(pdgId) == 6 || abs(pdgId) == 37 ",   # keep VIP(articles)s
        "keep abs(pdgId) == 310 && abs(eta) < 2.5 && pt > 1 ",                                                     # keep K0
# keep heavy flavour quarks for parton-based jet flavour
    "keep (4 <= abs(pdgId) <= 5) & (status = 2 || status = 11 || status = 71 || status = 72)",
# keep light-flavour quarks and gluons for parton-based jet flavour
    "keep (1 <= abs(pdgId) <= 3 || pdgId = 21) & (status = 2 || status = 11 || status = 71 || status = 72) && pt>5", 
# keep b and c hadrons for hadron-based jet flavour
    "keep (400 < abs(pdgId) < 600) || (4000 < abs(pdgId) < 6000)",
# additional c hadrons for jet fragmentation studies
    "keep abs(pdgId) = 10411 || abs(pdgId) = 10421 || abs(pdgId) = 10413 || abs(pdgId) = 10423 || abs(pdgId) = 20413 || abs(pdgId) = 20423 || abs(pdgId) = 10431 || abs(pdgId) = 10433 || abs(pdgId) = 20433", 
# additional b hadrons for jet fragmentation studies
    "keep abs(pdgId) = 10511 || abs(pdgId) = 10521 || abs(pdgId) = 10513 || abs(pdgId) = 10523 || abs(pdgId) = 20513 || abs(pdgId) = 20523 || abs(pdgId) = 10531 || abs(pdgId) = 10533 || abs(pdgId) = 20533 || abs(pdgId) = 10541 || abs(pdgId) = 10543 || abs(pdgId) = 20543", 
#keep SUSY particles
    "keep (1000001 <= abs(pdgId) <= 1000039 ) || ( 2000001 <= abs(pdgId) <= 2000015)",
# keep protons 
        "keep pdgId = 2212",
        "keep status == 3 || ( 21 <= status <= 29) || ( 11 <= status <= 19)",  #keep event summary (status=3 for pythia6, 21 <= status <= 29 for pythia8)

    )
)



##################### Tables for final output and docs ##########################
genParticleTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("finalGenParticles"),
    name= cms.string("GenPart"),
    doc = cms.string("interesting gen particles "),
    externalVariables = cms.PSet(
        iso = ExtVar(cms.InputTag("genIso"), float, doc="Isolation for leptons"),
    ),
    variables = cms.PSet(
         pt  = Var("pt",  float, precision=8),
         phi = Var("phi", float,precision=8),
         eta  = Var("eta",  float,precision=8),
         mass = Var("?!((abs(pdgId)>=1 && abs(pdgId)<=5) || (abs(pdgId)>=11 && abs(pdgId)<=16) || pdgId==21 || pdgId==111 || abs(pdgId)==211 || abs(pdgId)==421 || abs(pdgId)==411 || (pdgId==22 && mass<1))?mass:0", float,precision="?((abs(pdgId)==6 || abs(pdgId)>1000000) && statusFlags().isLastCopy())?20:8",doc="Mass stored for all particles with the exception of quarks (except top), leptons/neutrinos, photons with mass < 1 GeV, gluons, pi0(111), pi+(211), D0(421), and D+(411). For these particles, you can lookup the value from PDG."),
         pdgId  = Var("pdgId", int, doc="PDG id"),
         status  = Var("status", int, doc="Particle status. 1=stable"),
         genPartIdxMother = Var("?numberOfMothers>0?motherRef(0).key():-1", "int16", doc="index of the mother particle"),
         statusFlags = (Var(
            "statusFlags().isLastCopyBeforeFSR()                  * 16384 +"
            "statusFlags().isLastCopy()                           * 8192  +"
            "statusFlags().isFirstCopy()                          * 4096  +"
            "statusFlags().fromHardProcessBeforeFSR()             * 2048  +"
            "statusFlags().isDirectHardProcessTauDecayProduct()   * 1024  +"
            "statusFlags().isHardProcessTauDecayProduct()         * 512   +"
            "statusFlags().fromHardProcess()                      * 256   +"
            "statusFlags().isHardProcess()                        * 128   +"
            "statusFlags().isDirectHadronDecayProduct()           * 64    +"
            "statusFlags().isDirectPromptTauDecayProduct()        * 32    +"
            "statusFlags().isDirectTauDecayProduct()              * 16    +"
            "statusFlags().isPromptTauDecayProduct()              * 8     +"
            "statusFlags().isTauDecayProduct()                    * 4     +"
            "statusFlags().isDecayedLeptonHadron()                * 2     +"
            "statusFlags().isPrompt()                             * 1      ",
            "uint16", doc=("gen status flags stored bitwise, bits are: "
                "0 : isPrompt, "
                "1 : isDecayedLeptonHadron, "
                "2 : isTauDecayProduct, "
                "3 : isPromptTauDecayProduct, "
                "4 : isDirectTauDecayProduct, "
                "5 : isDirectPromptTauDecayProduct, "
                "6 : isDirectHadronDecayProduct, "
                "7 : isHardProcess, "
                "8 : fromHardProcess, "
                "9 : isHardProcessTauDecayProduct, "
                "10 : isDirectHardProcessTauDecayProduct, "
                "11 : fromHardProcessBeforeFSR, "
                "12 : isFirstCopy, "
                "13 : isLastCopy, "
                "14 : isLastCopyBeforeFSR, ")
            )),
    )
)

genIso = cms.EDProducer("GenPartIsoProducer",
                       src=cms.InputTag("finalGenParticles"),
        )

genParticleTask = cms.Task(finalGenParticles, genIso)
genParticleTablesTask = cms.Task(genParticleTable)
