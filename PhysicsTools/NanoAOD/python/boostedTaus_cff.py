import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *

##################### Import reusable funtions and objects from std taus ########
from PhysicsTools.NanoAOD.taus_cff import _tauId2WPMask,_tauId5WPMask,_tauId7WPMask,tausMCMatchLepTauForTable,tausMCMatchHadTauForTable,tauMCTable

##################### User floats producers, selectors ##########################


finalBoostedTaus = cms.EDFilter("PATTauRefSelector",
    src = cms.InputTag("slimmedTausBoosted"),
    cut = cms.string("pt > 40 && tauID('decayModeFindingNewDMs') && (tauID('byVVLooseIsolationMVArun2DBoldDMwLT') || tauID('byVVLooseIsolationMVArun2DBnewDMwLT'))")
)
run2_nanoAOD_106Xv2.toModify(finalBoostedTaus,
                             cut = "pt > 40 && tauID('decayModeFindingNewDMs') && (tauID('byVVLooseIsolationMVArun2017v2DBoldDMwLT2017') || tauID('byVVLooseIsolationMVArun2017v2DBoldDMdR0p3wLT2017') || tauID('byVVLooseIsolationMVArun2017v2DBnewDMwLT2017'))"
)

boostedTauTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("finalBoostedTaus"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name= cms.string("boostedTau"),
    doc = cms.string("slimmedBoostedTaus after basic selection (" + finalBoostedTaus.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the taus
    variables = cms.PSet() # PSet defined below in era dependent way
)
_boostedTauVarsBase = cms.PSet(P4Vars,
       charge = Var("charge", int, doc="electric charge"),
       jetIdx = Var("?hasUserCand('jet')?userCand('jet').key():-1", int, doc="index of the associated jet (-1 if none)"),
       decayMode = Var("decayMode()",int),
       leadTkPtOverTauPt = Var("leadChargedHadrCand.pt/pt ",float, doc="pt of the leading track divided by tau pt",precision=10),
       leadTkDeltaEta = Var("leadChargedHadrCand.eta - eta ",float, doc="eta of the leading track, minus tau eta",precision=8),
       leadTkDeltaPhi = Var("deltaPhi(leadChargedHadrCand.phi, phi) ",float, doc="phi of the leading track, minus tau phi",precision=8),

       rawIso = Var( "tauID('byCombinedIsolationDeltaBetaCorrRaw3Hits')", float, doc = "combined isolation (deltaBeta corrections)", precision=10),
       rawIsodR03 = Var( "(tauID('chargedIsoPtSumdR03')+max(0.,tauID('neutralIsoPtSumdR03')-0.072*tauID('puCorrPtSum')))", float, doc = "combined isolation (deltaBeta corrections, dR=0.3)", precision=10),
       chargedIso = Var( "tauID('chargedIsoPtSum')", float, doc = "charged isolation", precision=10),
       neutralIso = Var( "tauID('neutralIsoPtSum')", float, doc = "neutral (photon) isolation", precision=10),
       puCorr = Var( "tauID('puCorrPtSum')", float, doc = "pileup correction", precision=10),
       photonsOutsideSignalCone = Var( "tauID('photonPtSumOutsideSignalCone')", float, doc = "sum of photons outside signal cone", precision=10),
       idAntiMu = _tauId2WPMask("againstMuon%s3", doc= "Anti-muon discriminator V3: ")
)
#MVA 2017 v2 variables
_boostedTauVarsMVAIso = cms.PSet(
       rawMVAoldDM2017v2 = Var("tauID('byIsolationMVArun2DBoldDMwLTraw')",float, doc="byIsolationMVArun2DBoldDMwLT raw output discriminator (2017v2)",precision=10),
       rawMVAnewDM2017v2 = Var("tauID('byIsolationMVArun2DBnewDMwLTraw')",float,doc='byIsolationMVArun2DBnewDMwLT raw output discriminator (2017v2)',precision=10),
       idMVAnewDM2017v2 = _tauId7WPMask("by%sIsolationMVArun2DBnewDMwLT", doc="IsolationMVArun2DBnewDMwLT ID working point (2017v2)"),
       idMVAoldDM2017v2 = _tauId7WPMask("by%sIsolationMVArun2DBoldDMwLT",doc="IsolationMVArun2DBoldDMwLT ID working point (2017v2)"),
)
#MVA 2017 v2 dR<0.3 variables
_boostedTauVarsMVAIsoDr03 = cms.PSet(
       rawMVAoldDMdR032017v2 = Var("tauID('byIsolationMVArun2017v2DBoldDMdR0p3wLTraw2017')",float,doc='byIsolationMVArun2DBoldDMdR0p3wLT raw output discriminator (2017v2)'),
       idMVAoldDMdR032017v2 = _tauId7WPMask("by%sIsolationMVArun2017v2DBoldDMdR0p3wLT2017",doc="IsolationMVArun2DBoldDMdR0p3wLT ID working point (2017v2)")
)
#AntiEle MVA 2018 variables
_boostedTauVarsAntiEleMVA = cms.PSet(
       rawAntiEle2018 = Var("tauID('againstElectronMVA6Raw')", float, doc= "Anti-electron MVA discriminator V6 raw output discriminator (2018)", precision=10),
       rawAntiEleCat2018 = Var("tauID('againstElectronMVA6category')", int, doc="Anti-electron MVA discriminator V6 category (2018)"),
       idAntiEle2018 = _tauId5WPMask("againstElectron%sMVA6", doc= "Anti-electron MVA discriminator V6 (2018)")
)

boostedTauTable.variables = cms.PSet(
    _boostedTauVarsBase,
    _boostedTauVarsMVAIso,
    _boostedTauVarsAntiEleMVA
)
_boostedTauVarsWithDr03 = cms.PSet(
    _boostedTauVarsBase,
    _boostedTauVarsMVAIso,
    _boostedTauVarsMVAIsoDr03,
    _boostedTauVarsAntiEleMVA
)
run2_nanoAOD_106Xv2.toModify(boostedTauTable,
                             variables = _boostedTauVarsWithDr03
)
run2_nanoAOD_106Xv2.toModify(boostedTauTable.variables,
                             rawMVAoldDM2017v2 = Var("tauID('byIsolationMVArun2017v2DBoldDMwLTraw2017')",float, doc="byIsolationMVArun2DBoldDMwLT raw output discriminator (2017v2)",precision=10),
                             rawMVAnewDM2017v2 = Var("tauID('byIsolationMVArun2017v2DBnewDMwLTraw2017')",float,doc='byIsolationMVArun2DBnewDMwLT raw output discriminator (2017v2)',precision=10),
                             idMVAnewDM2017v2 = _tauId7WPMask("by%sIsolationMVArun2017v2DBnewDMwLT2017", doc="IsolationMVArun2DBnewDMwLT ID working point (2017v2)"),
                             idMVAoldDM2017v2 = _tauId7WPMask("by%sIsolationMVArun2017v2DBoldDMwLT2017",doc="IsolationMVArun2DBoldDMwLT ID working point (2017v2)"),
                             rawAntiEle2018 = Var("tauID('againstElectronMVA6Raw2018')", float, doc= "Anti-electron MVA discriminator V6 raw output discriminator (2018)", precision=10),
                             rawAntiEleCat2018 = Var("tauID('againstElectronMVA6category2018')", int, doc="Anti-electron MVA discriminator V6 category (2018)"),
                             idAntiEle2018 = _tauId5WPMask("againstElectron%sMVA62018", doc= "Anti-electron MVA discriminator V6 (2018)")
)

boostedTausMCMatchLepTauForTable = tausMCMatchLepTauForTable.clone(
    src = boostedTauTable.src
)

#This requires genVisTaus in taus_cff.py
boostedTausMCMatchHadTauForTable = tausMCMatchHadTauForTable.clone(
    src = boostedTauTable.src
)

boostedTauMCTable = tauMCTable.clone(
    src = boostedTauTable.src,
    mcMap = cms.InputTag("boostedTausMCMatchLepTauForTable"),
    mcMapVisTau = cms.InputTag("boostedTausMCMatchHadTauForTable"),
    objName = boostedTauTable.name,
)


boostedTauTask = cms.Task(finalBoostedTaus)
boostedTauTablesTask = cms.Task(boostedTauTable)
boostedTauMCTask = cms.Task(boostedTausMCMatchLepTauForTable,boostedTausMCMatchHadTauForTable,boostedTauMCTable)

#remove boosted tau from previous eras
_modifiers = (run2_miniAOD_80XLegacy | run2_nanoAOD_92X | run2_nanoAOD_94XMiniAODv1 | run2_nanoAOD_94X2016 | run2_nanoAOD_94XMiniAODv2 | run2_nanoAOD_102Xv1 | run2_nanoAOD_106Xv1 | run3_nanoAOD_122)
(_modifiers).toReplaceWith(boostedTauTask,cms.Task())
(_modifiers).toReplaceWith(boostedTauTablesTask,cms.Task())
(_modifiers).toReplaceWith(boostedTauMCTask,cms.Task())
