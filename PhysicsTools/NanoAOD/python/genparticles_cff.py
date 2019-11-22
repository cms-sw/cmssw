import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *



##################### User floats producers, selectors ##########################

finalGenParticles = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("prunedGenParticles"),
    select = cms.vstring(
	"drop *",
        "keep++ abs(pdgId) == 15 & (pt > 15 ||  isPromptDecayed() )",#  keep full tau decay chain for some taus
	#"drop status==1 & pt < 1", #drop soft stable particle in tau decay
        "keep+ abs(pdgId) == 15 ",  #  keep first gen decay product for all tau
        "+keep pdgId == 22 && status == 1 && (pt > 10 || isPromptFinalState())", # keep gamma above 10 GeV (or all prompt) and its first parent
	"+keep abs(pdgId) == 11 || abs(pdgId) == 13 || abs(pdgId) == 15", #keep leptons, with at most one mother back in the history
	"drop abs(pdgId)= 2212 && abs(pz) > 1000", #drop LHC protons accidentally added by previous keeps
        "keep (400 < abs(pdgId) < 600) || (4000 < abs(pdgId) < 6000)", #keep all B and C hadrons
        "keep abs(pdgId) == 12 || abs(pdgId) == 14 || abs(pdgId) == 16",   # keep neutrinos
	"keep status == 3 || (status > 20 && status < 30)", #keep matrix element summary
        "keep isHardProcess() ||  fromHardProcessDecayed()  || fromHardProcessFinalState() || (statusFlags().fromHardProcess() && statusFlags().isLastCopy())",  #keep event summary based on status flags
	"keep  (status > 70 && status < 80 && pt > 15) ", # keep high pt partons right before hadronization
        "keep abs(pdgId) == 23 || abs(pdgId) == 24 || abs(pdgId) == 25 || abs(pdgId) == 37 ",   # keep VIP(articles)s
        #"keep abs(pdgId) == 310 && abs(eta) < 2.5 && pt > 1 ",                                                     # keep K0
        "keep (1000001 <= abs(pdgId) <= 1000039 ) || ( 2000001 <= abs(pdgId) <= 2000015)", #keep SUSY fiction particles
 		
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
         pt  = Var("pt",  float, precision=8),
         phi = Var("phi", float,precision=8),
         eta  = Var("eta",  float,precision=8),
         mass = Var("?mass>10 || (pdgId==22 && mass > 1) || abs(pdgId)==24 || pdgId==23 || abs(pdgId)>1000000?mass:0", float,precision="?(abs(pdgId)==6 && statusFlags().isLastCopy())?20:8",doc="Mass stored for all particles with mass > 10 GeV and photons with mass > 1 GeV, plus W/Z and BSM particles. For other particles you can lookup from PDGID"),
         pdgId  = Var("pdgId", int, doc="PDG id"),
         status  = Var("status", int, doc="Particle status. 1=stable"),
         genPartIdxMother = Var("?numberOfMothers>0?motherRef(0).key():-1", int, doc="index of the mother particle"),
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
            int, doc=("gen status flags stored bitwise, bits are: "
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

genParticleSequence = cms.Sequence(finalGenParticles)
genParticleTables = cms.Sequence(genParticleTable)
