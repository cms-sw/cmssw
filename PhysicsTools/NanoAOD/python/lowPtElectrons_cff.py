import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *

################################################################################
# Modules
################################################################################

from RecoEgamma.EgammaTools.lowPtElectronModifier_cfi import lowPtElectronModifier
modifiedLowPtElectrons = cms.EDProducer(
    "ModifiedElectronProducer",
    src = cms.InputTag("slimmedLowPtElectrons"),
    modifierConfig = cms.PSet(
        modifications = cms.VPSet(lowPtElectronModifier)
    )
)

import PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi
updatedLowPtElectrons = cms.EDProducer(
    "PATElectronUpdater",
    src = cms.InputTag("modifiedLowPtElectrons"),
    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    computeMiniIso = cms.bool(True),
    fixDxySign = cms.bool(False),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParamsB = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsB,
    miniIsoParamsE = PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.miniIsoParamsE,
)

isoForLowPtEle = cms.EDProducer(
    "EleIsoValueMapProducer",
    src = cms.InputTag("updatedLowPtElectrons"),
    relative = cms.bool(True),
    rho_MiniIso = cms.InputTag("fixedGridRhoFastjetAll"),
    rho_PFIso = cms.InputTag("fixedGridRhoFastjetAll"),
    EAFile_MiniIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
    EAFile_PFIso = cms.FileInPath("RecoEgamma/ElectronIdentification/data/Fall17/effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt"),
)

updatedLowPtElectronsWithUserData = cms.EDProducer(
    "PATElectronUserDataEmbedder",
    src = cms.InputTag("updatedLowPtElectrons"),
    userFloats = cms.PSet(
        miniIsoChg = cms.InputTag("isoForLowPtEle:miniIsoChg"),
        miniIsoAll = cms.InputTag("isoForLowPtEle:miniIsoAll"),
    ),
    userIntFromBools = cms.PSet(),
    userInts = cms.PSet(),
    userCands = cms.PSet(),
)

finalLowPtElectrons = cms.EDFilter(
    "PATElectronRefSelector",
    src = cms.InputTag("updatedLowPtElectronsWithUserData"),
    cut = cms.string("pt > 1. && electronID('ID') > -0.25"),
)

################################################################################
# electronTable 
################################################################################

lowPtElectronTable = cms.EDProducer(
    "SimpleCandidateFlatTableProducer",
    src = cms.InputTag("linkedObjects","lowPtElectrons"),
    cut = cms.string(""),
    name= cms.string("LowPtElectron"),
    doc = cms.string("slimmedLowPtElectrons after basic selection (" + finalLowPtElectrons.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the electrons
    variables = cms.PSet(
        # Basic variables
        CandVars,
        # Overlaps with PF electron
        electronIdx = Var("?overlaps('electrons').size()>0?overlaps('electrons')[0].key():-1", int, doc="index of the overlapping PF electron (-1 if none)"),
        # BDT scores and WPs
        ID = Var("electronID('ID')",float,doc="ID, BDT (raw) score"),
        unbiased = Var("electronID('unbiased')",float,doc="ElectronSeed, pT- and dxy- agnostic BDT (raw) score"),
        ptbiased = Var("electronID('ptbiased')",float,doc="ElectronSeed, pT- and dxy- dependent BDT (raw) score"),
        # Isolation
        miniPFRelIso_chg = Var("userFloat('miniIsoChg')",float,
                               doc="mini PF relative isolation, charged component"),
        miniPFRelIso_all = Var("userFloat('miniIsoAll')",float,
                               doc="mini PF relative isolation, total (with scaled rho*EA PU corrections)"),
        # Conversions
        convVeto = Var("passConversionVeto()",bool,doc="pass conversion veto"),
        convWP = Var("userInt('convOpen')*1 + userInt('convLoose')*2 + userInt('convTight')*4",
                     int,doc="conversion flag bit map: 1=Veto, 2=Loose, 3=Tight"),
        convVtxRadius = Var("userFloat('convVtxRadius')",float,doc="conversion vertex radius (cm)",precision=7),
        # Tracking
        lostHits = Var("gsfTrack.hitPattern.numberOfLostHits('MISSING_INNER_HITS')","uint8",doc="number of missing inner hits"),
        # Cluster-related
        energyErr = Var("p4Error('P4_COMBINATION')",float,doc="energy error of the cluster-track combination",precision=6),
        deltaEtaSC = Var("superCluster().eta()-eta()",float,doc="delta eta (SC,ele) with sign",precision=10),
        r9 = Var("full5x5_r9()",float,doc="R9 of the SC, calculated with full 5x5 region",precision=10),
        sieie = Var("full5x5_sigmaIetaIeta()",float,doc="sigma_IetaIeta of the SC, calculated with full 5x5 region",precision=10),
        eInvMinusPInv = Var("(1-eSuperClusterOverP())/ecalEnergy()",float,doc="1/E_SC - 1/p_trk",precision=10),
        scEtOverPt = Var("(superCluster().energy()/(pt*cosh(superCluster().eta())))-1",float,doc="(SC energy)/pt-1",precision=8),
        hoe = Var("hadronicOverEm()",float,doc="H over E",precision=8),
        # Displacement
        dxy = Var("dB('PV2D')",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
        dxyErr = Var("edB('PV2D')",float,doc="dxy uncertainty, in cm",precision=6),
        dz = Var("dB('PVDZ')",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
        dzErr = Var("abs(edB('PVDZ'))",float,doc="dz uncertainty, in cm",precision=6),
        # Cross-referencing
        #jetIdx
        #photonIdx
    ),
)

################################################################################
# electronTable (MC)
################################################################################

# Depends on tautaggerForMatching being run in electrons_cff
matchingLowPtElecPhoton = cms.EDProducer(
    "GenJetGenPartMerger",
    srcJet =cms.InputTag("particleLevel:leptons"),
    srcPart=cms.InputTag("particleLevel:photons"),
    cut = cms.string(""),
    hasTauAnc=cms.InputTag("tautaggerForMatching"),
)

lowPtElectronsMCMatchForTableAlt = cms.EDProducer(
    "GenJetMatcherDRPtByDR",                # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = lowPtElectronTable.src,   # final reco collection
    matched     = cms.InputTag("matchingLowPtElecPhoton:merged"), # final mc-truth particle collection
    mcPdgId     = cms.vint32(11,22),        # one or more PDG ID (11 = el, 22 = pho); absolute values (see below)
    checkCharge = cms.bool(False),          # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(),
    maxDeltaR   = cms.double(0.3),          # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),          # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True), # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True), # False = just match input in order; True = pick lowest deltaR pair first
) 

lowPtElectronsMCMatchForTable = cms.EDProducer(
    "MCMatcher",                            # cut on deltaR, deltaPt/Pt; pick best by deltaR
    src         = lowPtElectronTable.src,   # final reco collection
    matched     = cms.InputTag("finalGenParticles"), # final mc-truth particle collection
    mcPdgId     = cms.vint32(11),           # one or more PDG ID (11 = ele); absolute values (see below)
    checkCharge = cms.bool(False),          # True = require RECO and MC objects to have the same charge
    mcStatus    = cms.vint32(1),            # PYTHIA status code (1 = stable, 2 = shower, 3 = hard scattering)
    maxDeltaR   = cms.double(0.3),          # Minimum deltaR for the match
    maxDPtRel   = cms.double(0.5),          # Minimum deltaPt/Pt for the match
    resolveAmbiguities    = cms.bool(True), # Forbid two RECO objects to match to the same GEN object
    resolveByMatchQuality = cms.bool(True), # False = just match input in order; True = pick lowest deltaR pair first
)

from PhysicsTools.NanoAOD.electrons_cff import electronMCTable
lowPtElectronMCTable = cms.EDProducer(
    "CandMCMatchTableProducer",
    src = lowPtElectronTable.src,
    mcMapDressedLep = cms.InputTag("lowPtElectronsMCMatchForTableAlt"),
    mcMap = cms.InputTag("lowPtElectronsMCMatchForTable"),
    mapTauAnc = cms.InputTag("matchingLowPtElecPhoton:hasTauAnc"),
    objName = lowPtElectronTable.name,
    objType = electronMCTable.objType,
    branchName = cms.string("genPart"),
    docString = cms.string("MC matching to status==1 electrons or photons"),
    genparticles = cms.InputTag("finalGenParticles"), 
)

################################################################################
# Tasks
################################################################################

lowPtElectronTask = cms.Task(modifiedLowPtElectrons,
                             updatedLowPtElectrons,
                             isoForLowPtEle,
                             updatedLowPtElectronsWithUserData,
                             finalLowPtElectrons)

lowPtElectronTablesTask = cms.Task(lowPtElectronTable)

lowPtElectronMCTask = cms.Task(
    matchingLowPtElecPhoton,
    lowPtElectronsMCMatchForTable,
    lowPtElectronsMCMatchForTableAlt,
    lowPtElectronMCTable)

################################################################################
# Modifiers
################################################################################

_modifiers = ( run2_miniAOD_80XLegacy |
               run2_nanoAOD_94XMiniAODv1 |
               run2_nanoAOD_94XMiniAODv2 |
               run2_nanoAOD_94X2016 |
               run2_nanoAOD_102Xv1 |
               run2_nanoAOD_106Xv1 )
(_modifiers).toReplaceWith(lowPtElectronTask,cms.Task())
(_modifiers).toReplaceWith(lowPtElectronTablesTask,cms.Task())
(_modifiers).toReplaceWith(lowPtElectronMCTask,cms.Task())

# To preserve "nano v9" functionality ...

from RecoEgamma.EgammaElectronProducers.lowPtGsfElectrons_cfi import lowPtRegressionModifier
run2_nanoAOD_106Xv2.toModify(modifiedLowPtElectrons.modifierConfig,
                             modifications = cms.VPSet(lowPtElectronModifier,
                                                       lowPtRegressionModifier))

run2_nanoAOD_106Xv2.toModify(updatedLowPtElectronsWithUserData.userFloats,
                             ID = cms.InputTag("lowPtPATElectronID"))

run2_nanoAOD_106Xv2.toModify(finalLowPtElectrons,
                             cut = "pt > 1. && userFloat('ID') > -0.25")

run2_nanoAOD_106Xv2.toModify(lowPtElectronTable.variables,
                             embeddedID = Var("electronID('ID')",float,doc="ID, BDT (raw) score"),
                             ID = Var("userFloat('ID')",float,doc="New ID, BDT (raw) score"))

from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronID_cfi import lowPtGsfElectronID
lowPtPATElectronID = lowPtGsfElectronID.clone(
    usePAT = True,
    electrons = "updatedLowPtElectrons",
    unbiased = "",
    ModelWeights = [
        'RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2020Nov28.root',
    ],
)

_lowPtElectronTask = cms.Task(lowPtPATElectronID)
_lowPtElectronTask.add(lowPtElectronTask.copy())
run2_nanoAOD_106Xv2.toReplaceWith(lowPtElectronTask,_lowPtElectronTask)
