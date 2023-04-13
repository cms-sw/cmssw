import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import run3_nanoAOD_124
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
from PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi import tauGenJetsSelectorAllHadrons

##################### Updated tau collection with MVA-based tau-Ids rerun #######
# Used only in some eras
from PhysicsTools.NanoAOD.taus_updatedMVAIds_cff import *

##################### User floats producers, selectors ##########################

# Original DeepTau v2p5 in 12_4_X doesn't include WPs in MINIAOD
# Import thresholds here to define WPs manually from raw scores
from RecoTauTag.RecoTau.tauIdWPsDefs import WORKING_POINTS_v2p5

finalTaus = cms.EDFilter("PATTauRefSelector",
    src = cms.InputTag("slimmedTaus"),
    cut = cms.string("pt > 18 && ((tauID('decayModeFindingNewDMs') > 0.5 && (tauID('byLooseCombinedIsolationDeltaBetaCorr3Hits') || (tauID('chargedIsoPtSumdR03')+max(0.,tauID('neutralIsoPtSumdR03')-0.072*tauID('puCorrPtSum'))<2.5) || tauID('byVVVLooseDeepTau2017v2p1VSjet') || tauID('byVVVLooseDeepTau2018v2p5VSjet'))) || (?isTauIDAvailable('byPNetVSjetraw')?tauID('byPNetVSjetraw'):-1) > {})".format(0.05))
)

run3_nanoAOD_124.toModify(
    finalTaus,
    cut = cms.string("pt > 18 && ((tauID('decayModeFindingNewDMs') > 0.5 && (tauID('byLooseCombinedIsolationDeltaBetaCorr3Hits') || (tauID('chargedIsoPtSumdR03')+max(0.,tauID('neutralIsoPtSumdR03')-0.072*tauID('puCorrPtSum'))<2.5) || tauID('byVVVLooseDeepTau2017v2p1VSjet') || (tauID('byDeepTau2018v2p5VSjetraw') > {}))) || (?isTauIDAvailable('byPNetVSjetraw')?tauID('byPNetVSjetraw'):-1) > {})".format(WORKING_POINTS_v2p5["jet"]["VVVLoose"],0.05))
)

##################### Tables for final output and docs ##########################
def _tauIdWPMask(pattern, choices, doc="", from_raw=False, wp_thrs=None):
    if from_raw:
        assert wp_thrs is not None, "wp_thrs argument in _tauIdWPMask() is None, expect it to be dict-like"

        var_definition = []
        for wp_name in choices:
            if not isinstance(wp_thrs[wp_name], float):
                raise TypeError("Threshold for WP=%s is not a float number." % wp_name)
            wp_definition = "test_bit(tauID('{}')-{}+1,0)".format(pattern, wp_thrs[wp_name])
            var_definition.append(wp_definition)
        var_definition = " + ".join(var_definition)
        var_definition = ("?isTauIDAvailable('%s')?(" % pattern) + var_definition + "):0"
    else:
        var_definition = " + ".join(["tauID('%s')" % (pattern % c) for c in choices])
        var_definition = ("?isTauIDAvailable('%s')?(" % (pattern % choices[0])) + var_definition + "):0"

    doc = doc + ": "+", ".join(["%d = %s" % (i,c) for (i,c) in enumerate(choices, start=1)])
    return Var(var_definition, "uint8", doc=doc)


tauTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","taus"),
    name= cms.string("Tau"),
    doc = cms.string("slimmedTaus after basic selection (" + finalTaus.cut.value()+")")
)

_tauVarsBase = cms.PSet(P4Vars,
       charge = Var("charge", "int16", doc="electric charge"),
       jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", "int16", doc="index of the associated jet (-1 if none)"),
       eleIdx = Var("?overlaps('electrons').size()>0?overlaps('electrons')[0].key():-1", "int16", doc="index of first matching electron"),
       muIdx = Var("?overlaps('muons').size()>0?overlaps('muons')[0].key():-1", "int16", doc="index of first matching muon"),
       svIdx1 = Var("?overlaps('vertices').size()>0?overlaps('vertices')[0].key():-1", "int16", doc="index of first matching secondary vertex"),
       svIdx2 = Var("?overlaps('vertices').size()>1?overlaps('vertices')[1].key():-1", "int16", doc="index of second matching secondary vertex"),
       nSVs = Var("?hasOverlaps('vertices')?overlaps('vertices').size():0", "uint8", doc="number of secondary vertices in the tau"),
       decayMode = Var("decayMode()", "uint8"),
       idDecayModeOldDMs = Var("(?isTauIDAvailable('decayModeFinding')?tauID('decayModeFinding'):-1) > 0", bool),
       idDecayModeNewDMs = Var("(?isTauIDAvailable('decayModeFindingNewDMs')?tauID('decayModeFindingNewDMs'):-1) > 0", bool),
       leadTkPtOverTauPt = Var("?leadChargedHadrCand.isNonnull()?leadChargedHadrCand.pt/pt:1",float, doc="pt of the leading track divided by tau pt",precision=10),
       leadTkDeltaEta = Var("?leadChargedHadrCand.isNonnull()?(leadChargedHadrCand.eta - eta):0",float, doc="eta of the leading track, minus tau eta",precision=8),
       leadTkDeltaPhi = Var("?leadChargedHadrCand.isNonnull()?deltaPhi(leadChargedHadrCand.phi, phi):0",float, doc="phi of the leading track, minus tau phi",precision=8),

       dxy = Var("?leadChargedHadrCand.isNonnull()?leadChargedHadrCand().dxy():0",float, doc="d_{xy} of lead track with respect to PV, in cm (with sign)",precision=10),
       dz = Var("?leadChargedHadrCand.isNonnull()?leadChargedHadrCand().dz():0",float, doc="d_{z} of lead track with respect to PV, in cm (with sign)",precision=14),

       # these are too many, we may have to suppress some
       rawIso = Var("?isTauIDAvailable('byCombinedIsolationDeltaBetaCorrRaw3Hits')?tauID('byCombinedIsolationDeltaBetaCorrRaw3Hits'):-1", float, doc = "combined isolation (deltaBeta corrections)", precision=10),
       rawIsodR03 = Var("?isTauIDAvailable('chargedIsoPtSumdR03')?(tauID('chargedIsoPtSumdR03')+max(0.,tauID('neutralIsoPtSumdR03')-0.072*tauID('puCorrPtSum'))):-1", float, doc = "combined isolation (deltaBeta corrections, dR=0.3)", precision=10),
       chargedIso = Var("?isTauIDAvailable('chargedIsoPtSum')?tauID('chargedIsoPtSum'):-1", float, doc = "charged isolation", precision=10),
       neutralIso = Var("?isTauIDAvailable('neutralIsoPtSum')?tauID('neutralIsoPtSum'):-1", float, doc = "neutral (photon) isolation", precision=10),
       puCorr = Var("?isTauIDAvailable('puCorrPtSum')?tauID('puCorrPtSum'):-1", float, doc = "pileup correction", precision=10),
       photonsOutsideSignalCone = Var("?isTauIDAvailable('photonPtSumOutsideSignalCone')?tauID('photonPtSumOutsideSignalCone'):-1", float, doc = "sum of photons outside signal cone", precision=10),

       idAntiMu = _tauIdWPMask("againstMuon%s3", choices=("Loose","Tight"), doc= "Anti-muon discriminator V3: "),
       idAntiEleDeadECal = Var("(?isTauIDAvailable('againstElectronDeadECAL')?tauID('againstElectronDeadECAL'):-1) > 0", bool, doc = "Anti-electron dead-ECal discriminator"),

)

_deepTauVars2017v2p1 = cms.PSet(
    rawDeepTau2017v2p1VSe = Var("?isTauIDAvailable('byDeepTau2017v2p1VSeraw')?tauID('byDeepTau2017v2p1VSeraw'):-1", float, doc="byDeepTau2017v2p1VSe raw output discriminator (deepTau2017v2p1)", precision=10),
    rawDeepTau2017v2p1VSmu = Var("?isTauIDAvailable('byDeepTau2017v2p1VSmuraw')?tauID('byDeepTau2017v2p1VSmuraw'):-1", float, doc="byDeepTau2017v2p1VSmu raw output discriminator (deepTau2017v2p1)", precision=10),
    rawDeepTau2017v2p1VSjet = Var("?isTauIDAvailable('byDeepTau2017v2p1VSjetraw')?tauID('byDeepTau2017v2p1VSjetraw'):-1", float, doc="byDeepTau2017v2p1VSjet raw output discriminator (deepTau2017v2p1)", precision=10),
    idDeepTau2017v2p1VSe = _tauIdWPMask("by%sDeepTau2017v2p1VSe",
                                            choices=("VVVLoose","VVLoose","VLoose","Loose","Medium","Tight","VTight","VVTight"),
                                            doc="byDeepTau2017v2p1VSe ID working points (deepTau2017v2p1)"),
    idDeepTau2017v2p1VSmu = _tauIdWPMask("by%sDeepTau2017v2p1VSmu",
                                            choices=("VLoose", "Loose", "Medium", "Tight"),
                                            doc="byDeepTau2017v2p1VSmu ID working points (deepTau2017v2p1)"),
    idDeepTau2017v2p1VSjet = _tauIdWPMask("by%sDeepTau2017v2p1VSjet",
                                            choices=("VVVLoose","VVLoose","VLoose","Loose","Medium","Tight","VTight","VVTight"),
                                            doc="byDeepTau2017v2p1VSjet ID working points (deepTau2017v2p1)"),
)
_deepTauVars2018v2p5 = cms.PSet(
    rawDeepTau2018v2p5VSe = Var("?isTauIDAvailable('byDeepTau2018v2p5VSeraw')?tauID('byDeepTau2018v2p5VSeraw'):-1", float, doc="byDeepTau2018v2p5VSe raw output discriminator (deepTau2018v2p5)", precision=10),
    rawDeepTau2018v2p5VSmu = Var("?isTauIDAvailable('byDeepTau2018v2p5VSmuraw')?tauID('byDeepTau2018v2p5VSmuraw'):-1", float, doc="byDeepTau2018v2p5VSmu raw output discriminator (deepTau2018v2p5)", precision=10),
    rawDeepTau2018v2p5VSjet = Var("?isTauIDAvailable('byDeepTau2018v2p5VSjetraw')?tauID('byDeepTau2018v2p5VSjetraw'):-1", float, doc="byDeepTau2018v2p5VSjet raw output discriminator (deepTau2018v2p5)", precision=10),
    idDeepTau2018v2p5VSe = _tauIdWPMask("by%sDeepTau2018v2p5VSe",
                                            choices=("VVVLoose","VVLoose","VLoose","Loose","Medium","Tight","VTight","VVTight"),
                                            doc="byDeepTau2018v2p5VSe ID working points (deepTau2018v2p5)"),
    idDeepTau2018v2p5VSmu = _tauIdWPMask("by%sDeepTau2018v2p5VSmu",
                                            choices=("VLoose", "Loose", "Medium", "Tight"),
                                            doc="byDeepTau2018v2p5VSmu ID working points (deepTau2018v2p5)"),
    idDeepTau2018v2p5VSjet = _tauIdWPMask("by%sDeepTau2018v2p5VSjet",
                                            choices=("VVVLoose","VVLoose","VLoose","Loose","Medium","Tight","VTight","VVTight"),
                                            doc="byDeepTau2018v2p5VSjet ID working points (deepTau2018v2p5)"),
)

_pnet2023 = cms.PSet(
    decayModePNet = Var("?isTauIDAvailable('byPNetDecayMode')?tauID('byPNetDecayMode'):-1", "int16",doc="decay mode of the highest tau score of ParticleNet"),
    rawPNetVSe = Var("?isTauIDAvailable('byPNetVSeraw')?tauID('byPNetVSeraw'):-1", float, doc="raw output of ParticleNetVsE discriminator (PNet 2023)", precision=10),
    rawPNetVSmu = Var("?isTauIDAvailable('byPNetVSmuraw')?tauID('byPNetVSmuraw'):-1", float, doc="raw output of ParticleNetVsMu discriminator (PNet 2023)", precision=10),
    rawPNetVSjet = Var("?isTauIDAvailable('byPNetVSjetraw')?tauID('byPNetVSjetraw'):-1", float, doc="raw output of ParticleNetVsJet discriminator (PNet 2023)", precision=10),
    ptCorrPNet = Var("?isTauIDAvailable('byPNetPtCorr')?tauID('byPNetPtCorr'):1", float, doc="pt correction (PNet 2023)", precision=10),
)

_variablesMiniV2 = cms.PSet(
    _tauVarsBase,
    _deepTauVars2017v2p1,
    _deepTauVars2018v2p5,
    _pnet2023
)

tauTable.variables = _variablesMiniV2

run3_nanoAOD_124.toModify(
    tauTable.variables,
    idDeepTau2018v2p5VSe = _tauIdWPMask("byDeepTau2018v2p5VSeraw",
                 choices=("VVVLoose","VVLoose","VLoose","Loose","Medium","Tight","VTight","VVTight"),
                 doc="byDeepTau2018v2p5VSe ID working points (deepTau2018v2p5)",
                 from_raw=True, wp_thrs=WORKING_POINTS_v2p5["e"]),
    idDeepTau2018v2p5VSmu = _tauIdWPMask("byDeepTau2018v2p5VSmuraw",
                 choices=("VLoose", "Loose", "Medium", "Tight"),
                 doc="byDeepTau2018v2p5VSmu ID working points (deepTau2018v2p5)",
                 from_raw=True, wp_thrs=WORKING_POINTS_v2p5["mu"]),
    idDeepTau2018v2p5VSjet = _tauIdWPMask("byDeepTau2018v2p5VSjetraw",
                 choices=("VVVLoose","VVLoose","VLoose","Loose","Medium","Tight","VTight","VVTight"),
                 doc="byDeepTau2018v2p5VSjet ID working points (deepTau2018v2p5)",
                 from_raw=True, wp_thrs=WORKING_POINTS_v2p5["jet"])
)


tauGenJetsForNano = tauGenJets.clone(
    GenParticles = "finalGenParticles",
    includeNeutrinos = False
)

tauGenJetsSelectorAllHadronsForNano = tauGenJetsSelectorAllHadrons.clone(
    src = "tauGenJetsForNano"
)

genVisTaus = cms.EDProducer("GenVisTauProducer",
    src = cms.InputTag("tauGenJetsSelectorAllHadronsForNano"),
    srcGenParticles = cms.InputTag("finalGenParticles")
)

genVisTauTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("genVisTaus"),
    cut = cms.string("pt > 10."),
    name = cms.string("GenVisTau"),
    doc = cms.string("gen hadronic taus "),
    variables = cms.PSet(
         pt = Var("pt", float,precision=8),
         phi = Var("phi", float,precision=8),
         eta = Var("eta", float,precision=8),
         mass = Var("mass", float,precision=8),
	 charge = Var("charge", "int16"),
	 status = Var("status", "uint8", doc="Hadronic tau decay mode. 0=OneProng0PiZero, 1=OneProng1PiZero, 2=OneProng2PiZero, 10=ThreeProng0PiZero, 11=ThreeProng1PiZero, 15=Other"),
	 genPartIdxMother = Var("?numberOfMothers>0?motherRef(0).key():-1", "int16", doc="index of the mother particle"),
    )
)

tausMCMatchLepTauForTable = cms.EDProducer("MCMatcher",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = tauTable.src,                 # final reco collection
    matched     = cms.InputTag("finalGenParticles"), # final mc-truth particle collection
    mcPdgId     = cms.vint32(11,13),            # one or more PDG ID (11 = electron, 13 = muon); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(),                 # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),              # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),              # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),     # False = just match input in order; True = pick lowest deltaR pair first
)

tausMCMatchHadTauForTable = cms.EDProducer("MCMatcher",  # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = tauTable.src,                 # final reco collection
    matched     = cms.InputTag("genVisTaus"),   # generator level hadronic tau decays
    mcPdgId     = cms.vint32(15),               # one or more PDG ID (15 = tau); absolute values (see below)
    checkCharge = cms.bool(False),              # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(),                 # CV: no *not* require certain status code for matching (status code corresponds to decay mode for hadronic tau decays)
    maxDeltaR   = cms.double(0.3),              # Maximum deltaR for the match
    maxDPtRel   = cms.double(1.),               # Maximum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True),     # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True),     # False = just match input in order; True = pick lowest deltaR pair first
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


tauTask = cms.Task(finalTaus)
tauTablesTask = cms.Task(tauTable)

genTauTask = cms.Task(tauGenJetsForNano,tauGenJetsSelectorAllHadronsForNano,genVisTaus,genVisTauTable)
tauMCTask = cms.Task(genTauTask,tausMCMatchLepTauForTable,tausMCMatchHadTauForTable,tauMCTable)

