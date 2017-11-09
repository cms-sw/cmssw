import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
from PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi import tauGenJetsSelectorAllHadrons 


##################### User floats producers, selectors ##########################


finalTaus = cms.EDFilter("PATTauRefSelector",
    src = cms.InputTag("slimmedTaus"),
    cut = cms.string("pt > 18 && tauID('decayModeFindingNewDMs') && (tauID('byLooseCombinedIsolationDeltaBetaCorr3Hits') || tauID('byVLooseIsolationMVArun2v1DBoldDMwLT') || tauID('byVLooseIsolationMVArun2v1DBnewDMwLT') || tauID('byVLooseIsolationMVArun2v1DBdR03oldDMwLT'))")
)

##################### Tables for final output and docs ##########################
def _tauIdWPMask(pattern, choices, doc=""):
    return Var(" + ".join(["%d * tauID('%s')" % (pow(2,i), pattern % c) for (i,c) in enumerate(choices)]), "uint8", 
               doc=doc+": bitmask "+", ".join(["%d = %s" % (pow(2,i),c) for (i,c) in enumerate(choices)]))
def _tauId2WPMask(pattern,doc):
    return _tauIdWPMask(pattern,choices=("Loose","Tight"),doc=doc)
def _tauId5WPMask(pattern,doc):
    return _tauIdWPMask(pattern,choices=("VLoose","Loose","Medium","Tight","VTight"),doc=doc)
def _tauId6WPMask(pattern,doc):
    return _tauIdWPMask(pattern,choices=("VLoose","Loose","Medium","Tight","VTight","VVTight"),doc=doc)


tauTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("linkedObjects","taus"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name= cms.string("Tau"),
    doc = cms.string("slimmedTaus after basic selection (" + finalTaus.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the taus
    variables = cms.PSet(P4Vars,
       charge = Var("charge", int, doc="electric charge"),                  
       jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", int, doc="index of the associated jet (-1 if none)"),
       decayMode = Var("decayMode()",int),
       idDecayMode = Var("tauID('decayModeFinding')", bool),
       idDecayModeNewDMs = Var("tauID('decayModeFindingNewDMs')", bool),

       leadTkPtOverTauPt = Var("leadChargedHadrCand.pt/pt ",float, doc="pt of the leading track divided by tau pt",precision=10),
       leadTkDeltaEta = Var("leadChargedHadrCand.eta - eta ",float, doc="eta of the leading track, minus tau eta",precision=8),
       leadTkDeltaPhi = Var("deltaPhi(leadChargedHadrCand.phi, phi) ",float, doc="phi of the leading track, minus tau phi",precision=8),

       dxy = Var("dxy()",float, doc="d_{xy} of lead track with respect to PV, in cm (with sign)",precision=10),
       dz = Var("leadChargedHadrCand().dz() ",float, doc="d_{z} of lead track with respect to PV, in cm (with sign)",precision=14),

       # these are too many, we may have to suppress some
       rawIso = Var( "tauID('byCombinedIsolationDeltaBetaCorrRaw3Hits')", float, doc = "combined isolation (deltaBeta corrections)", precision=10),
       chargedIso = Var( "tauID('chargedIsoPtSum')", float, doc = "charged isolation", precision=10),
       neutralIso = Var( "tauID('neutralIsoPtSum')", float, doc = "neutral (photon) isolation", precision=10),
       puCorr = Var( "tauID('puCorrPtSum')", float, doc = "pileup correction", precision=10),
       footprintCorr = Var( "tauID('footprintCorrection')", float, doc = "footprint correction", precision=10),
       photonsOutsideSignalCone = Var( "tauID('photonPtSumOutsideSignalCone')", float, doc = "sum of photons outside signal cone", precision=10),
                         
       rawMVAnewDM = Var( "tauID('byIsolationMVArun2v1DBnewDMwLTraw')",float, doc="byIsolationMVArun2v1DBnewDMwLT raw output discriminator",precision=10),
       rawMVAoldDM = Var( "tauID('byIsolationMVArun2v1DBoldDMwLTraw')",float, doc="byIsolationMVArun2v1DBoldDMwLT raw output discriminator",precision=10),
       rawMVAoldDMdR03 = Var( "tauID('byIsolationMVArun2v1DBdR03oldDMwLTraw')",float, doc="byIsolationMVArun2v1DBdR03oldDMwLT raw output discriminator",precision=10),
       rawAntiEle = Var("tauID('againstElectronMVA6Raw')", float, doc= "Anti-electron MVA discriminator V6 raw output discriminator", precision=10),
       rawAntiEleCat = Var("tauID('againstElectronMVA6category')", int, doc="Anti-electron MVA discriminator V6 category"),
       
       idAntiMu = _tauId2WPMask("againstMuon%s3", doc= "Anti-muon discriminator V3: "),
       idAntiEle = _tauId5WPMask("againstElectron%sMVA6", doc= "Anti-electron MVA discriminator V6"),                         
       idMVAnewDM = _tauId6WPMask( "by%sIsolationMVArun2v1DBnewDMwLT", doc="IsolationMVArun2v1DBnewDMwLT ID working point"),
       idMVAoldDM = _tauId6WPMask( "by%sIsolationMVArun2v1DBoldDMwLT", doc="IsolationMVArun2v1DBoldDMwLT ID working point"),
       idMVAoldDMdR03 = _tauId6WPMask( "by%sIsolationMVArun2v1DBdR03oldDMwLT", doc="IsolationMVArun2v1DBdR03oldDMwLT ID working point"),
    

#   isoCI3hit = Var(  "tauID("byCombinedIsolationDeltaBetaCorrRaw3Hits")" doc="byCombinedIsolationDeltaBetaCorrRaw3Hits raw output discriminator"),
#   photonOutsideSigCone = Var( "tauID("photonPtSumOutsideSignalCone")" doc="photonPtSumOutsideSignalCone raw output discriminator"),


    )
)

tauGenJets.GenParticles = cms.InputTag("prunedGenParticles")
tauGenJets.includeNeutrinos = cms.bool(False)

genVisTaus = cms.EDProducer("GenVisTauProducer",
    src = cms.InputTag("tauGenJetsSelectorAllHadrons"),         
    srcGenParticles = cms.InputTag("prunedGenParticles")
)

genVisTauTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("genVisTaus"),
    cut = cms.string("pt > 10."),
    name= cms.string("GenVisTau"),
    doc = cms.string("gen hadronic taus "),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for generator level hadronic tau decays
    variables = cms.PSet(
         pt  = Var("pt",  float,precision=8),
         phi = Var("phi", float,precision=8),
         eta  = Var("eta",  float,precision=8),
	 pdgId  = Var("pdgId", int, doc="PDG id"),
	 status  = Var("status", int, doc="Hadronic tau decay mode. 0=OneProng0PiZero, 1=OneProng1PiZero, 2=OneProng2PiZero, 10=ThreeProng0PiZero, 11=ThreeProng1PiZero, 15=Other"),
	 genPartIdxMother = Var("?numberOfMothers>0?motherRef(0).key():-1", int, doc="index of the mother particle"),
    )
)

tausMCMatchLepTauForTable = cms.EDProducer("MCMatcher",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = tauTable.src,                 # final reco collection
    matched     = cms.InputTag("finalGenParticles"), # final mc-truth particle collection
    mcPdgId     = cms.vint32(11,13),            # one or more PDG ID (11 = electron, 13 = muon); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),                # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),    # False = just match input in order; True = pick lowest deltaR pair first
)

tausMCMatchHadTauForTable = cms.EDProducer("MCMatcher",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = tauTable.src,                 # final reco collection
    matched     = cms.InputTag("genVisTaus"), # generator level hadronic tau decays
    mcPdgId     = cms.vint32(15),                    # one or more PDG ID (15 = tau); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(),                 # CV: no *not* require certain status code for matching (status code corresponds to decay mode for hadronic tau decays)
    maxDeltaR   = cms.double(0.3),              # Maximum deltaR for the match
    maxDPtRel   = cms.double(1.),               # Maximum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),    # False = just match input in order; True = pick lowest deltaR pair first
)

tauMCTable = cms.EDProducer("CandMCMatchTableProducer",
    src = tauTable.src,
    mcMap = cms.InputTag("tausMCMatchLepTauForTable"),
    mcMapVisTau = cms.InputTag("tausMCMatchHadTauForTable"),                         
    objName = tauTable.name,
    objType = tauTable.name, #cms.string("Tau"),
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==2 taus"),
)


tauSequence = cms.Sequence(finalTaus)
tauTables = cms.Sequence(tauTable)
tauMC = cms.Sequence(tauGenJets + tauGenJetsSelectorAllHadrons + genVisTaus + genVisTauTable + tausMCMatchLepTauForTable + tausMCMatchHadTauForTable + tauMCTable)

