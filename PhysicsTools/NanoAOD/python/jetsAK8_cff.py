import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simplePATJetFlatTableProducer_cfi import simplePATJetFlatTableProducer

from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
# Note: Safe to always add 'L2L3Residual' as MC contains dummy L2L3Residual corrections (always set to 1)
#      (cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#CMSSW_7_6_4_and_above )
jetCorrFactorsAK8 = patJetCorrFactors.clone(src='slimmedJetsAK8',
    levels = cms.vstring('L1FastJet',
        'L2Relative',
        'L3Absolute',
        'L2L3Residual'),
    payload = cms.string('AK8PFPuppi'),
    primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

from  PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import *
updatedJetsAK8 = updatedPatJets.clone(
    addBTagInfo=False,
    jetSource='slimmedJetsAK8',
    jetCorrFactorsSource=cms.VInputTag(cms.InputTag("jetCorrFactorsAK8") ),
)

updatedJetsAK8WithUserData = cms.EDProducer("PATJetUserDataEmbedder",
    src = cms.InputTag("updatedJetsAK8"),
    userFloats = cms.PSet(),
    userInts = cms.PSet(),
)

finalJetsAK8 = cms.EDFilter("PATJetRefSelector",
    src = cms.InputTag("updatedJetsAK8WithUserData"),
    cut = cms.string("pt > 170")
)


lepInAK8JetVars = cms.EDProducer("LepInJetProducer",
    src = cms.InputTag("updatedJetsAK8WithUserData"),
    srcEle = cms.InputTag("finalElectrons"),
    srcMu = cms.InputTag("finalMuons")
)

fatJetTable = simplePATJetFlatTableProducer.clone(
    src = cms.InputTag("finalJetsAK8"),
    cut = cms.string(" pt > 170"), #probably already applied in miniaod
    name = cms.string("FatJet"),
    doc  = cms.string("slimmedJetsAK8, i.e. ak8 fat jets for boosted analysis"),
    variables = cms.PSet(P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        tau1 = Var("userFloat('NjettinessAK8Puppi:tau1')",float, doc="Nsubjettiness (1 axis)",precision=10),
        tau2 = Var("userFloat('NjettinessAK8Puppi:tau2')",float, doc="Nsubjettiness (2 axis)",precision=10),
        tau3 = Var("userFloat('NjettinessAK8Puppi:tau3')",float, doc="Nsubjettiness (3 axis)",precision=10),
        tau4 = Var("userFloat('NjettinessAK8Puppi:tau4')",float, doc="Nsubjettiness (4 axis)",precision=10),
        n2b1 = Var("?hasUserFloat('nb1AK8PuppiSoftDrop:ecfN2')?userFloat('nb1AK8PuppiSoftDrop:ecfN2'):-99999.", float, doc="N2 with beta=1 (for jets with raw pT>250 GeV)", precision=10),
        n3b1 = Var("?hasUserFloat('nb1AK8PuppiSoftDrop:ecfN3')?userFloat('nb1AK8PuppiSoftDrop:ecfN3'):-99999.", float, doc="N3 with beta=1 (for jets with raw pT>250 GeV)", precision=10),
        msoftdrop = Var("groomedMass('SoftDropPuppi')",float, doc="Corrected soft drop mass with PUPPI",precision=10),
        globalParT3_Xbb = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXbb')",float,doc="Mass-decorrelated GlobalParT-3 X->bb score. Note: For sig vs bkg (e.g. bkg=QCD) tagging, use sig/(sig+bkg) to construct the discriminator",precision=10),
        globalParT3_Xcc = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXcc')",float,doc="Mass-decorrelated GlobalParT-3 X->cc score",precision=10),
        globalParT3_Xcs = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXcs')",float,doc="Mass-decorrelated GlobalParT-3 X->cs score",precision=10),
        globalParT3_Xqq = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXqq')",float,doc="Mass-decorrelated GlobalParT-3 X->qq (ss/dd/uu) score",precision=10),
        globalParT3_Xtauhtaue = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXtauhtaue')",float,doc="Mass-decorrelated GlobalParT-3 X->tauhtaue score",precision=10),
        globalParT3_Xtauhtaum = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXtauhtaum')",float,doc="Mass-decorrelated GlobalParT-3 X->tauhtaum score",precision=10),
        globalParT3_Xtauhtauh = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXtauhtauh')",float,doc="Mass-decorrelated GlobalParT-3 X->tauhtauh score",precision=10),
        globalParT3_XWW4q = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXWW4q')",float,doc="Mass-decorrelated GlobalParT-3 X->WW4q score",precision=10),
        globalParT3_XWW3q = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXWW3q')",float,doc="Mass-decorrelated GlobalParT-3 X->WW3q score",precision=10),
        globalParT3_XWWqqev = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXWWqqev')",float,doc="Mass-decorrelated GlobalParT-3 X->WWqqev score",precision=10),
        globalParT3_XWWqqmv = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXWWqqmv')",float,doc="Mass-decorrelated GlobalParT-3 X->WWqqmv score",precision=10),
        globalParT3_TopbWqq = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probTopbWqq')",float,doc="Mass-decorrelated GlobalParT-3 Top->bWqq score",precision=10),
        globalParT3_TopbWq = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probTopbWq')",float,doc="Mass-decorrelated GlobalParT-3 Top->bWq score",precision=10),
        globalParT3_TopbWev = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probTopbWev')",float,doc="Mass-decorrelated GlobalParT-3 Top->bWev score",precision=10),
        globalParT3_TopbWmv = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probTopbWmv')",float,doc="Mass-decorrelated GlobalParT-3 Top->bWmv score",precision=10),
        globalParT3_TopbWtauhv = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probTopbWtauhv')",float,doc="Mass-decorrelated GlobalParT-3 Top->bWtauhv score",precision=10),
        globalParT3_QCD = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probQCD')",float,doc="Mass-decorrelated GlobalParT-3 QCD score.",precision=10),
        globalParT3_WvsQCD = Var("?bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXqq')+bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXcs')+bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probQCD')>0?(bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXqq')+bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXcs'))/(bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXqq')+bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probXcs')+bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probQCD')):-1",
            float,doc="Mass-decorrelated GlobalParT-3 (Xqq+Xcs/Xqq+Xcs+QCD) binarized score.",precision=10),
        globalParT3_massCorrX2p = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:massCorrX2p')",float,doc="GlobalParT-3 mass regression corrector with respect to the original jet mass, optimised for resonance 2-prong (bb/cc/cs/ss/qq) jets. Use (massCorrX2p * mass * (1 - rawFactor)) to get the regressed mass",precision=10),
        globalParT3_massCorrGeneric = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:massCorrGeneric')",float,doc="GlobalParT-3 mass regression corrector with respect to the original jet mass, optimised for generic jet cases. Use (massCorrGeneric * mass * (1 - rawFactor)) to get the regressed mass",precision=10),
        globalParT3_withMassTopvsQCD = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probWithMassTopvsQCD')",float,doc="GlobalParT-3 tagger (w/mass) Top vs QCD discriminator",precision=10),
        globalParT3_withMassWvsQCD = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probWithMassWvsQCD')",float,doc="GlobalParT-3 tagger (w/mass) W vs QCD discriminator",precision=10),
        globalParT3_withMassZvsQCD = Var("bDiscriminator('pfGlobalParticleTransformerAK8JetTags:probWithMassZvsQCD')",float,doc="GlobalParT-3 tagger (w/mass) Z vs QCD discriminator",precision=10),
        particleNetWithMass_QCD = Var("bDiscriminator('pfParticleNetJetTags:probQCDbb')+bDiscriminator('pfParticleNetJetTags:probQCDcc')+bDiscriminator('pfParticleNetJetTags:probQCDb')+bDiscriminator('pfParticleNetJetTags:probQCDc')+bDiscriminator('pfParticleNetJetTags:probQCDothers')",float,doc="ParticleNet tagger (w/ mass) QCD(bb,cc,b,c,others) sum",precision=10),
        particleNetWithMass_TvsQCD = Var("bDiscriminator('pfParticleNetDiscriminatorsJetTags:TvsQCD')",float,doc="ParticleNet tagger (w/ mass) top vs QCD discriminator",precision=10),
        particleNetWithMass_WvsQCD = Var("bDiscriminator('pfParticleNetDiscriminatorsJetTags:WvsQCD')",float,doc="ParticleNet tagger (w/ mass) W vs QCD discriminator",precision=10),
        particleNetWithMass_ZvsQCD = Var("bDiscriminator('pfParticleNetDiscriminatorsJetTags:ZvsQCD')",float,doc="ParticleNet tagger (w/ mass) Z vs QCD discriminator",precision=10),
        particleNetWithMass_H4qvsQCD = Var("bDiscriminator('pfParticleNetDiscriminatorsJetTags:H4qvsQCD')",float,doc="ParticleNet tagger (w/ mass) H(->VV->qqqq) vs QCD discriminator",precision=10),
        particleNetWithMass_HbbvsQCD = Var("bDiscriminator('pfParticleNetDiscriminatorsJetTags:HbbvsQCD')",float,doc="ParticleNet tagger (w/mass) H(->bb) vs QCD discriminator",precision=10),
        particleNetWithMass_HccvsQCD = Var("bDiscriminator('pfParticleNetDiscriminatorsJetTags:HccvsQCD')",float,doc="ParticleNet tagger (w/mass) H(->cc) vs QCD discriminator",precision=10),
        particleNet_QCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8JetTags:probQCD2hf')+bDiscriminator('pfParticleNetFromMiniAODAK8JetTags:probQCD1hf')+bDiscriminator('pfParticleNetFromMiniAODAK8JetTags:probQCD0hf')",float,doc="ParticleNet tagger QCD(0+1+2HF) sum",precision=10),
        particleNet_QCD2HF = Var("bDiscriminator('pfParticleNetFromMiniAODAK8JetTags:probQCD2hf')",float,doc="ParticleNet tagger QCD 2 HF (b/c) score",precision=10),
        particleNet_QCD1HF = Var("bDiscriminator('pfParticleNetFromMiniAODAK8JetTags:probQCD1hf')",float,doc="ParticleNet tagger QCD 1 HF (b/c) score",precision=10),
        particleNet_QCD0HF = Var("bDiscriminator('pfParticleNetFromMiniAODAK8JetTags:probQCD0hf')",float,doc="ParticleNet tagger QCD 0 HF (b/c) score",precision=10),
        particleNet_massCorr = Var("bDiscriminator('pfParticleNetFromMiniAODAK8JetTags:masscorr')",float,doc="ParticleNet mass regression, relative correction to JEC-corrected jet mass (no softdrop)",precision=10),
        particleNet_XbbVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HbbvsQCD')",float,doc="ParticleNet X->bb vs. QCD score: Xbb/(Xbb+QCD)",precision=10),
        particleNet_XccVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HccvsQCD')",float,doc="ParticleNet X->cc vs. QCD score: Xcc/(Xcc+QCD)",precision=10),
        particleNet_XqqVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HqqvsQCD')",float,doc="ParticleNet X->qq (uds) vs. QCD score: Xqq/(Xqq+QCD)",precision=10),
        particleNet_XggVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HggvsQCD')",float,doc="ParticleNet X->gg vs. QCD score: Xgg/(Xgg+QCD)",precision=10),
        particleNet_XttVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HttvsQCD')",float,doc="ParticleNet X->tau_h tau_h vs. QCD score: Xtt/(Xtt+QCD)",precision=10),
        particleNet_XtmVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HtmvsQCD')",float,doc="ParticleNet X->mu tau_h vs. QCD score: Xtm/(Xtm+QCD)",precision=10),
        particleNet_XteVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:HtevsQCD')",float,doc="ParticleNet X->e tau_h vs. QCD score: Xte/(Xte+QCD)",precision=10),
        particleNet_WVsQCD = Var("bDiscriminator('pfParticleNetFromMiniAODAK8DiscriminatorsJetTags:WvsQCD')",float,doc="ParticleNet W->qq vs. QCD score: Xqq+Xcc/(Xqq+Xcc+QCD)",precision=10),
        particleNetLegacy_mass = Var("bDiscriminator('pfParticleNetMassRegressionJetTags:mass')",float,doc="ParticleNet Legacy Run-2 mass regression",precision=10),
        particleNetLegacy_Xbb = Var("bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probXbb')",float,doc="Mass-decorrelated ParticleNet Legacy Run-2 tagger raw X->bb score. For X->bb vs QCD tagging, use Xbb/(Xbb+QCD)",precision=10),
        particleNetLegacy_Xcc = Var("bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probXcc')",float,doc="Mass-decorrelated ParticleNet Legacy Run-2 tagger raw X->cc score. For X->cc vs QCD tagging, use Xcc/(Xcc+QCD)",precision=10),
        particleNetLegacy_Xqq = Var("bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probXqq')",float,doc="Mass-decorrelated ParticleNet Legacy Run-2 tagger raw X->qq (uds) score. For X->qq vs QCD tagging, use Xqq/(Xqq+QCD). For W vs QCD tagging, use (Xcc+Xqq)/(Xcc+Xqq+QCD)",precision=10),
        particleNetLegacy_QCD = Var("bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probQCDbb')+bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probQCDcc')+bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probQCDb')+bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probQCDc')+bDiscriminator('pfMassDecorrelatedParticleNetJetTags:probQCDothers')",float,doc="Mass-decorrelated ParticleNet Legacy Run-2 tagger raw QCD score",precision=10),
        subJetIdx1 = Var("?nSubjetCollections()>0 && subjets('SoftDropPuppi').size()>0?subjets('SoftDropPuppi')[0].key():-1", "int16",
            doc="index of first subjet"),
        subJetIdx2 = Var("?nSubjetCollections()>0 && subjets('SoftDropPuppi').size()>1?subjets('SoftDropPuppi')[1].key():-1", "int16",
            doc="index of second subjet"),
        nConstituents = Var("numberOfDaughters()","uint8",doc="Number of particles in the jet"),
        chMultiplicity = Var("?isPFJet()?chargedMultiplicity():-1","int16",doc="(Puppi-weighted) Number of charged particles in the jet"),
        neMultiplicity = Var("?isPFJet()?neutralMultiplicity():-1","int16",doc="(Puppi-weighted) Number of neutral particles in the jet"),
        chHEF = Var("?isPFJet()?chargedHadronEnergyFraction():-1", float, doc="charged Hadron Energy Fraction", precision=10),
        neHEF = Var("?isPFJet()?neutralHadronEnergyFraction():-1", float, doc="neutral Hadron Energy Fraction", precision=10),
        chEmEF = Var("?isPFJet()?chargedEmEnergyFraction():-1", float, doc="charged Electromagnetic Energy Fraction", precision=10),
        neEmEF = Var("?isPFJet()?neutralEmEnergyFraction():-1", float, doc="neutral Electromagnetic Energy Fraction", precision=10),
        hfHEF = Var("?isPFJet()?HFHadronEnergyFraction():-1",float,doc="hadronic Energy Fraction in HF",precision=10),
        hfEmEF = Var("?isPFJet()?HFEMEnergyFraction():-1",float,doc="electromagnetic Energy Fraction in HF",precision=10),
        muEF = Var("?isPFJet()?muonEnergyFraction():-1", float, doc="muon Energy Fraction", precision=10),
    ),
    externalVariables = cms.PSet(
        lsf3 = ExtVar(cms.InputTag("lepInAK8JetVars:lsf3"),float, doc="Lepton Subjet Fraction (3 subjets)",precision=10),
        muonIdx3SJ = ExtVar(cms.InputTag("lepInAK8JetVars:muIdx3SJ"),"int16", doc="index of muon matched to jet"),
        electronIdx3SJ = ExtVar(cms.InputTag("lepInAK8JetVars:eleIdx3SJ"),"int16",doc="index of electron matched to jet"),
    )
)

##############################################################
## DeepInfoAK8:Start
## - To be used in nanoAOD_customizeCommon() in nano_cff.py
###############################################################
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
def nanoAOD_addDeepInfoAK8(process, addDeepBTag, addDeepBoostedJet, addDeepDoubleX, addDeepDoubleXV2, addParticleNetMassLegacy, addParticleNetLegacy, addParticleNet, addGlobalParT, jecPayload):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag to AK8 jets")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb']
    if addDeepBoostedJet:
        print("Updating process to run DeepBoostedJet on datasets before 103X")
        from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import _pfDeepBoostedJetTagsAll as pfDeepBoostedJetTagsAll
        _btagDiscriminators += pfDeepBoostedJetTagsAll
    if addGlobalParT:
        print("Updating process to run GlobalParT")
        from RecoBTag.ONNXRuntime.pfGlobalParticleTransformerAK8_cff import _pfGlobalParticleTransformerAK8JetTagsAll as pfGlobalParticleTransformerAK8JetTagsAll
        _btagDiscriminators += pfGlobalParticleTransformerAK8JetTagsAll
    if addParticleNet:
        print("Updating process to run ParticleNet joint classification and mass regression")
        from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsAll as pfParticleNetFromMiniAODAK8JetTagsAll
        _btagDiscriminators += pfParticleNetFromMiniAODAK8JetTagsAll
    if addParticleNetLegacy:
        from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll
        _btagDiscriminators += _pfParticleNetJetTagsAll
    if addParticleNetMassLegacy:
        from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassRegressionOutputs
        _btagDiscriminators += _pfParticleNetMassRegressionOutputs
    if addDeepDoubleX:
        print("Updating process to run DeepDoubleX on datasets before 104X")
        _btagDiscriminators += ['pfDeepDoubleBvLJetTags:probHbb', \
            'pfDeepDoubleCvLJetTags:probHcc', \
            'pfDeepDoubleCvBJetTags:probHcc', \
            'pfMassIndependentDeepDoubleBvLJetTags:probHbb', 'pfMassIndependentDeepDoubleCvLJetTags:probHcc', 'pfMassIndependentDeepDoubleCvBJetTags:probHcc']
    if addDeepDoubleXV2:
        print("Updating process to run DeepDoubleXv2 on datasets before 11X")
        _btagDiscriminators += [
            'pfMassIndependentDeepDoubleBvLV2JetTags:probHbb',
            'pfMassIndependentDeepDoubleCvLV2JetTags:probHcc',
            'pfMassIndependentDeepDoubleCvBV2JetTags:probHcc'
            ]
    if len(_btagDiscriminators)==0: return process
    print("Will recalculate the following discriminators on AK8 jets: "+", ".join(_btagDiscriminators))
    updateJetCollection(
       process,
       jetSource=cms.InputTag('slimmedJetsAK8', processName=cms.InputTag.skipCurrentProcess()),
       pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
       svSource = cms.InputTag('slimmedSecondaryVertices'),
       rParam = 0.8,
       jetCorrections = (jecPayload.value(), cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute', 'L2L3Residual']), 'None'),
       btagDiscriminators = _btagDiscriminators,
       postfix='AK8WithDeepInfo',
       printWarning = False
    )

    from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask
    task = getPatAlgosToolsTask(process)
    addToProcessAndTask("slimmedJetsAK8", process.selectedUpdatedPatJetsAK8WithDeepInfo.clone(), process, task)
    del process.selectedUpdatedPatJetsAK8WithDeepInfo

    return process

nanoAOD_addDeepInfoAK8_switch = cms.PSet(
    nanoAOD_addDeepBTag_switch = cms.untracked.bool(False),
    nanoAOD_addDeepBoostedJet_switch = cms.untracked.bool(False),
    nanoAOD_addDeepDoubleX_switch = cms.untracked.bool(False),
    nanoAOD_addDeepDoubleXV2_switch = cms.untracked.bool(False),
    nanoAOD_addParticleNetMassLegacy_switch = cms.untracked.bool(False),
    nanoAOD_addParticleNetLegacy_switch = cms.untracked.bool(False),
    nanoAOD_addParticleNet_switch = cms.untracked.bool(False),
    nanoAOD_addGlobalParT_switch = cms.untracked.bool(False),
    jecPayload = cms.untracked.string('AK8PFPuppi')
)

################################################
## DeepInfoAK8:End
#################################################

subJetTable = simplePATJetFlatTableProducer.clone(
    src = cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked","SubJets"),
    name = cms.string("SubJet"),
    doc  = cms.string("slimmedJetsAK8PFPuppiSoftDropPacked::SubJets, i.e. soft-drop subjets for ak8 fat jets for boosted analysis"),
    variables = cms.PSet(P4Vars,
        btagDeepFlavB = Var("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')",float,doc="DeepJet b+bb+lepb tag discriminator",precision=10),
        btagUParTAK4B = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll'):-1",float,precision=10,doc="UnifiedParT b vs. udscg"),
        UParTAK4RegPtRawCorr = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr'):-1",float,precision=10,doc="UnifiedParT universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT"),
        UParTAK4RegPtRawCorrNeutrino = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu'):-1",float,precision=10,doc="UnifiedParT universal flavor-aware pT regression neutrino correction, relative to visible. Correction relative to raw jet pT"),
        UParTAK4RegPtRawRes = Var("?(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow'))>0?0.5*(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow')):-1",float,precision=10,doc="UnifiedParT universal flavor-aware jet pT resolution estimator, (q84 - q16)/2"),
        UParTAK4V1RegPtRawCorr = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptcorr')>0?bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptcorr'):-1",float,precision=10,doc="UnifiedParT V1 universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT"),
        UParTAK4V1RegPtRawCorrNeutrino = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptnu')>0?bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptnu'):-1",float,precision=10,doc="UnifiedParT V1 universal flavor-aware pT regression neutrino correction, relative to visible. Correction relative to raw jet pT"),
        UParTAK4V1RegPtRawRes = Var("?(bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptreshigh')+bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptreslow'))>0?0.5*(bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptreshigh')-bDiscriminator('pfUnifiedParticleTransformerAK4V1JetTags:ptreslow')):-1",float,precision=10,doc="UnifiedParT V1 universal flavor-aware jet pT resolution estimator, (q84 - q16)/2"),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        tau1 = Var("userFloat('NjettinessAK8Subjets:tau1')",float, doc="Nsubjettiness (1 axis)",precision=10),
        tau2 = Var("userFloat('NjettinessAK8Subjets:tau2')",float, doc="Nsubjettiness (2 axis)",precision=10),
        tau3 = Var("userFloat('NjettinessAK8Subjets:tau3')",float, doc="Nsubjettiness (3 axis)",precision=10),
        tau4 = Var("userFloat('NjettinessAK8Subjets:tau4')",float, doc="Nsubjettiness (4 axis)",precision=10),
        n2b1 = Var("userFloat('nb1AK8PuppiSoftDropSubjets:ecfN2')", float, doc="N2 with beta=1", precision=10),
        n3b1 = Var("userFloat('nb1AK8PuppiSoftDropSubjets:ecfN3')", float, doc="N3 with beta=1", precision=10),
    )
)

#jets are not as precise as muons
fatJetTable.variables.pt.precision=10
subJetTable.variables.pt.precision=10

jetAK8UserDataTask = cms.Task()
jetAK8Task = cms.Task(jetCorrFactorsAK8,updatedJetsAK8,jetAK8UserDataTask,updatedJetsAK8WithUserData,finalJetsAK8)

#after lepton collections have been run
jetAK8LepTask = cms.Task(lepInAK8JetVars)

jetAK8TablesTask = cms.Task(fatJetTable,subJetTable)
