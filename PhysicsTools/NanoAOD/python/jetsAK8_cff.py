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

run2_nanoAOD_ANY.toModify(
    fatJetTable.variables,
    btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
    # Remove for V9
    chMultiplicity = None,
    neMultiplicity = None,
    chHEF = None,
    neHEF = None,
    chEmEF = None,
    neEmEF = None,
    muEF = None
)

(run2_nanoAOD_106Xv2).toModify(
    fatJetTable.variables,
    # Restore taggers that were decommisionned for Run-3
    btagDeepB = Var("?(bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'))>=0?bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'):-1",float,doc="DeepCSV b+bb tag discriminator",precision=10),
    btagHbb = Var("bDiscriminator('pfBoostedDoubleSecondaryVertexAK8BJetTags')",float,doc="Higgs to BB tagger discriminator",precision=10),
    btagDDBvLV2 = Var("bDiscriminator('pfMassIndependentDeepDoubleBvLV2JetTags:probHbb')",float,doc="DeepDoubleX V2(mass-decorrelated) discriminator for H(Z)->bb vs QCD",precision=10),
    btagDDCvLV2 = Var("bDiscriminator('pfMassIndependentDeepDoubleCvLV2JetTags:probHcc')",float,doc="DeepDoubleX V2 (mass-decorrelated) discriminator for H(Z)->cc vs QCD",precision=10),
    btagDDCvBV2 = Var("bDiscriminator('pfMassIndependentDeepDoubleCvBV2JetTags:probHcc')",float,doc="DeepDoubleX V2 (mass-decorrelated) discriminator for H(Z)->cc vs H(Z)->bb",precision=10),
    deepTag_TvsQCD = Var("bDiscriminator('pfDeepBoostedDiscriminatorsJetTags:TvsQCD')",float,doc="DeepBoostedJet tagger top vs QCD discriminator",precision=10),
    deepTag_WvsQCD = Var("bDiscriminator('pfDeepBoostedDiscriminatorsJetTags:WvsQCD')",float,doc="DeepBoostedJet tagger W vs QCD discriminator",precision=10),
    deepTag_ZvsQCD = Var("bDiscriminator('pfDeepBoostedDiscriminatorsJetTags:ZvsQCD')",float,doc="DeepBoostedJet tagger Z vs QCD discriminator",precision=10),
    deepTag_H = Var("bDiscriminator('pfDeepBoostedJetTags:probHbb')+bDiscriminator('pfDeepBoostedJetTags:probHcc')+bDiscriminator('pfDeepBoostedJetTags:probHqqqq')",float,doc="DeepBoostedJet tagger H(bb,cc,4q) sum",precision=10),
    deepTag_QCD = Var("bDiscriminator('pfDeepBoostedJetTags:probQCDbb')+bDiscriminator('pfDeepBoostedJetTags:probQCDcc')+bDiscriminator('pfDeepBoostedJetTags:probQCDb')+bDiscriminator('pfDeepBoostedJetTags:probQCDc')+bDiscriminator('pfDeepBoostedJetTags:probQCDothers')",float,doc="DeepBoostedJet tagger QCD(bb,cc,b,c,others) sum",precision=10),
    deepTag_QCDothers = Var("bDiscriminator('pfDeepBoostedJetTags:probQCDothers')",float,doc="DeepBoostedJet tagger QCDothers value",precision=10),
    deepTagMD_TvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger top vs QCD discriminator",precision=10),
    deepTagMD_WvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger W vs QCD discriminator",precision=10),
    deepTagMD_ZvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger Z vs QCD discriminator",precision=10),
    deepTagMD_ZHbbvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger Z/H->bb vs QCD discriminator",precision=10),
    deepTagMD_ZbbvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZbbvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger Z->bb vs QCD discriminator",precision=10),
    deepTagMD_HbbvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:HbbvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger H->bb vs QCD discriminator",precision=10),
    deepTagMD_ZHccvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHccvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger Z/H->cc vs QCD discriminator",precision=10),
    deepTagMD_H4qvsQCD = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:H4qvsQCD')",float,doc="Mass-decorrelated DeepBoostedJet tagger H->4q vs QCD discriminator",precision=10),
    deepTagMD_bbvsLight = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight')",float,doc="Mass-decorrelated DeepBoostedJet tagger Z/H/gluon->bb vs light flavour discriminator",precision=10),
    deepTagMD_ccvsLight = Var("bDiscriminator('pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ccvsLight')",float,doc="Mass-decorrelated DeepBoostedJet tagger Z/H/gluon->cc vs light flavour discriminator",precision=10),
)

##############################################################
## DeepInfoAK8:Start
## - To be used in nanoAOD_customizeCommon() in nano_cff.py
###############################################################
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
def nanoAOD_addDeepInfoAK8(process, addDeepBTag, addDeepBoostedJet, addDeepDoubleX, addDeepDoubleXV2, addParticleNetMassLegacy, addParticleNet, jecPayload):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag to AK8 jets")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb']
    if addDeepBoostedJet:
        print("Updating process to run DeepBoostedJet on datasets before 103X")
        from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import _pfDeepBoostedJetTagsAll as pfDeepBoostedJetTagsAll
        _btagDiscriminators += pfDeepBoostedJetTagsAll
    if addParticleNet:
        print("Updating process to run ParticleNet joint classification and mass regression")
        from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsAll as pfParticleNetFromMiniAODAK8JetTagsAll
        _btagDiscriminators += pfParticleNetFromMiniAODAK8JetTagsAll
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
       jetSource = cms.InputTag('slimmedJetsAK8'),
       pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
       svSource = cms.InputTag('slimmedSecondaryVertices'),
       rParam = 0.8,
       jetCorrections = (jecPayload.value(), cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute', 'L2L3Residual']), 'None'),
       btagDiscriminators = _btagDiscriminators,
       postfix='AK8WithDeepInfo',
       printWarning = False
    )
    process.jetCorrFactorsAK8.src="selectedUpdatedPatJetsAK8WithDeepInfo"
    process.updatedJetsAK8.jetSource="selectedUpdatedPatJetsAK8WithDeepInfo"
    return process

nanoAOD_addDeepInfoAK8_switch = cms.PSet(
    nanoAOD_addDeepBTag_switch = cms.untracked.bool(False),
    nanoAOD_addDeepBoostedJet_switch = cms.untracked.bool(False),
    nanoAOD_addDeepDoubleX_switch = cms.untracked.bool(False),
    nanoAOD_addDeepDoubleXV2_switch = cms.untracked.bool(False),
    nanoAOD_addParticleNetMassLegacy_switch = cms.untracked.bool(False),
    nanoAOD_addParticleNet_switch = cms.untracked.bool(False),
    jecPayload = cms.untracked.string('AK8PFPuppi')
)


# ParticleNet legacy jet tagger is already in 106Xv2 MINIAOD,
# add ParticleNet legacy mass regression and new combined tagger + mass regression
run2_nanoAOD_106Xv2.toModify(
    nanoAOD_addDeepInfoAK8_switch,
    nanoAOD_addParticleNetMassLegacy_switch = True,
    nanoAOD_addParticleNet_switch = True,
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

run2_nanoAOD_ANY.toModify(
    subJetTable.variables,
    btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10)
)

(run2_nanoAOD_106Xv2).toModify(
    subJetTable.variables,
    area = None,
)

run3_nanoAOD_pre142X.toModify(
    subJetTable.variables,
    btagDeepFlavB = None,
    btagUParTAK4B = None,
    btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="DeepCSV b+bb tag discriminator",precision=10),
)


#jets are not as precise as muons
fatJetTable.variables.pt.precision=10
subJetTable.variables.pt.precision=10

##############################################################
# AK8 constituents
###############################################################
finalJetsAK8PFConstituents = cms.EDProducer("PatJetConstituentPtrSelector",
    src = fatJetTable.src,
    cut = cms.string("abs(eta) <= 2.5")
)

finalJetsPFConstituents = cms.EDProducer("PackedCandidatePtrMerger",
    src = cms.VInputTag(cms.InputTag("finalJetsAK8PFConstituents", "constituents")),
    skipNulls = cms.bool(True),
    warnOnSkip = cms.bool(True)
)

pfCandidatesTable = cms.EDProducer("SimplePATCandidateFlatTableProducer",
    src = cms.InputTag("finalJetsPFConstituents"),
    cut = cms.string(""),
    name = cms.string("PFCand"),
    doc = cms.string("PF candidate constituents of AK8 puppi jets (FatJet) with |eta| <= 2.5."),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt * puppiWeight()", float, doc="Puppi-weighted pt", precision=10),
        mass = Var("mass * puppiWeight()", float, doc="Puppi-weighted mass", precision=10),
        eta = Var("eta", float, precision=12),
        phi = Var("phi", float, precision=12),
        pdgId  = Var("pdgId", int, doc="PF candidate type (+/-211 = ChgHad, 130 = NeuHad, 22 = Photon, +/-11 = Electron, +/-13 = Muon, 1 = HFHad, 2 = HFEM)")
    )
)

finalJetsAK8ConstituentsTable = cms.EDProducer("SimplePatJetConstituentTableProducer",
    name = cms.string(fatJetTable.name.value()+"PFCand"),
    candIdxName = cms.string("PFCandIdx"),
    candIdxDoc = cms.string("Index in the PFCand table"),
    jets = fatJetTable.src,
    candidates = pfCandidatesTable.src,
    jetCut = fatJetTable.cut
)

jetAK8UserDataTask = cms.Task()
jetAK8Task = cms.Task(jetCorrFactorsAK8,updatedJetsAK8,jetAK8UserDataTask,updatedJetsAK8WithUserData,finalJetsAK8,finalJetsAK8PFConstituents,finalJetsPFConstituents)

#after lepton collections have been run
jetAK8LepTask = cms.Task(lepInAK8JetVars)

jetAK8TablesTask = cms.Task(fatJetTable,subJetTable,pfCandidatesTable,finalJetsAK8ConstituentsTable)

