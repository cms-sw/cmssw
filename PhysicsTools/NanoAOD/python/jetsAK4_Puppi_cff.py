import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.simplePATJetFlatTableProducer_cfi import simplePATJetFlatTableProducer

##################### User floats producers, selectors ##########################

from  PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
# Note: Safe to always add 'L2L3Residual' as MC contains dummy L2L3Residual corrections (always set to 1)
#      (cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#CMSSW_7_6_4_and_above )
jetPuppiCorrFactorsNano = patJetCorrFactors.clone(src='slimmedJetsPuppi',
    levels = cms.vstring('L1FastJet',
        'L2Relative',
        'L3Absolute',
        'L2L3Residual'),
    payload = cms.string('AK4PFPuppi'),
    primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

from  PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import *

updatedJetsPuppi = updatedPatJets.clone(
    addBTagInfo=False,
    jetSource='slimmedJetsPuppi',
    jetCorrFactorsSource=cms.VInputTag(cms.InputTag("jetPuppiCorrFactorsNano") ),
)

tightJetPuppiId = cms.EDProducer("PatJetIDValueMapProducer",
    filterParams=cms.PSet(
        version = cms.string('RUN3PUPPIruns2022FGruns2023CD'),
        quality = cms.string('TIGHT'),
    ),
    src = cms.InputTag("updatedJetsPuppi")
)
tightJetPuppiIdLepVeto = cms.EDProducer("PatJetIDValueMapProducer",
    filterParams=cms.PSet(
        version = cms.string('RUN3PUPPIruns2022FGruns2023CD'),
        quality = cms.string('TIGHTLEPVETO'),
    ),
    src = cms.InputTag("updatedJetsPuppi")
)

run2_jme_2016.toModify(
    tightJetPuppiId.filterParams, version = "RUN2UL16PUPPI"
).toModify(
    tightJetPuppiIdLepVeto.filterParams, version = "RUN2UL16PUPPI"
)

(run2_jme_2017 | run2_jme_2018 | run3_nanoAOD_122 | run3_nanoAOD_124).toModify(
    tightJetPuppiId.filterParams, version = "RUN2ULPUPPI"
).toModify(
    tightJetPuppiIdLepVeto.filterParams, version = "RUN2ULPUPPI"
)

run3_jme_Winter22runsBCDEprompt.toModify(
    tightJetPuppiId.filterParams, version = "RUN3PUPPIruns2022BCDEprompt"
).toModify(
    tightJetPuppiIdLepVeto.filterParams, version = "RUN3PUPPIruns2022BCDEprompt"
)

#HF shower shape recomputation
from RecoJets.JetProducers.hfJetShowerShape_cfi import hfJetShowerShape
hfJetPuppiShowerShapeforNanoAOD = hfJetShowerShape.clone(jets="updatedJetsPuppi",vertices="offlineSlimmedPrimaryVertices")

updatedJetsPuppiWithUserData = cms.EDProducer("PATJetUserDataEmbedder",
    src = cms.InputTag("updatedJetsPuppi"),
    userFloats = cms.PSet(
        hfsigmaEtaEta = cms.InputTag('hfJetPuppiShowerShapeforNanoAOD:sigmaEtaEta'),
        hfsigmaPhiPhi = cms.InputTag('hfJetPuppiShowerShapeforNanoAOD:sigmaPhiPhi'),
    ),
    userInts = cms.PSet(
        tightId = cms.InputTag("tightJetPuppiId"),
        tightIdLepVeto = cms.InputTag("tightJetPuppiIdLepVeto"),
        hfcentralEtaStripSize = cms.InputTag('hfJetPuppiShowerShapeforNanoAOD:centralEtaStripSize'),
        hfadjacentEtaStripsSize = cms.InputTag('hfJetPuppiShowerShapeforNanoAOD:adjacentEtaStripsSize'),
    ),
)

finalJetsPuppi = cms.EDFilter("PATJetRefSelector",
    src = cms.InputTag("updatedJetsPuppiWithUserData"),
    cut = cms.string("pt > 15")
)

##################### Tables for final output and docs ##########################
jetPuppiTable = simplePATJetFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","jets"),
    name = cms.string("Jet"),
    doc  = cms.string("slimmedJetsPuppi, i.e. ak4 PFJets Puppi with JECs applied, after basic selection (" + finalJetsPuppi.cut.value()+")"),
    externalVariables = cms.PSet(),
    variables = cms.PSet(P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        nMuons = Var("?hasOverlaps('muons')?overlaps('muons').size():0", "uint8", doc="number of muons in the jet"),
        muonIdx1 = Var("?overlaps('muons').size()>0?overlaps('muons')[0].key():-1", "int16", doc="index of first matching muon"),
        muonIdx2 = Var("?overlaps('muons').size()>1?overlaps('muons')[1].key():-1", "int16", doc="index of second matching muon"),
        electronIdx1 = Var("?overlaps('electrons').size()>0?overlaps('electrons')[0].key():-1", "int16", doc="index of first matching electron"),
        electronIdx2 = Var("?overlaps('electrons').size()>1?overlaps('electrons')[1].key():-1", "int16", doc="index of second matching electron"),
        nElectrons = Var("?hasOverlaps('electrons')?overlaps('electrons').size():0", "uint8", doc="number of electrons in the jet"),
        svIdx1 = Var("?overlaps('vertices').size()>0?overlaps('vertices')[0].key():-1", "int16", doc="index of first matching secondary vertex"),
        svIdx2 = Var("?overlaps('vertices').size()>1?overlaps('vertices')[1].key():-1", "int16", doc="index of second matching secondary vertex"),
        nSVs = Var("?hasOverlaps('vertices')?overlaps('vertices').size():0", "uint8", doc="number of secondary vertices in the jet"),
        btagDeepFlavB = Var("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')",float,doc="DeepJet b+bb+lepb tag discriminator",precision=10),
        btagDeepFlavCvL = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1",float,doc="DeepJet c vs uds+g discriminator",precision=10),
        btagDeepFlavCvB = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1",float,doc="DeepJet c vs b+bb+lepb discriminator",precision=10),
        btagDeepFlavQG = Var("?(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1",float,doc="DeepJet g vs uds discriminator",precision=10),
        btagPNetB = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:BvsAll')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:BvsAll'):-1",float,precision=10,doc="ParticleNet b vs. udscg"),
        btagPNetCvNotB = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probc')/(1.-bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:probb')):-1",float,precision=10,doc="ParticleNet C vs notB"),
        btagPNetCvL = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:CvsL')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:CvsL'):-1",float,precision=10,doc="ParticleNet c vs. udsg"),
        btagPNetCvB = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:CvsB')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:CvsB'):-1",float,precision=10,doc="ParticleNet c vs. b"),
        btagPNetQvG = Var("?abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:QvsG'):bDiscriminator('pfParticleNetFromMiniAODAK4PuppiForwardDiscriminatorsJetTags:QvsG')",float,precision=10,doc="ParticleNet q (udsbc) vs. g"),
        btagPNetTauVJet = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:TauVsJet')>0?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralDiscriminatorsJetTags:TauVsJet'):-1",float,precision=10,doc="ParticleNet tau vs. jet"),
        PNetRegPtRawCorr = Var("?abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptcorr'):bDiscriminator('pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptcorr')",float,precision=10,doc="ParticleNet universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT"),
        PNetRegPtRawCorrNeutrino = Var("?abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptnu'):bDiscriminator('pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptnu')",float,precision=10,doc="ParticleNet universal flavor-aware pT regression neutrino correction, relative to visible. To apply full regression, multiply raw jet pT by both PNetRegPtRawCorr and PNetRegPtRawCorrNeutrino."),
        PNetRegPtRawRes = Var("?abs(eta())<2.5?0.5*(bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptreshigh')-bDiscriminator('pfParticleNetFromMiniAODAK4PuppiCentralJetTags:ptreslow')):0.5*(bDiscriminator('pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptreshigh')-bDiscriminator('pfParticleNetFromMiniAODAK4PuppiForwardJetTags:ptreslow'))",float,precision=10,doc="ParticleNet universal flavor-aware jet pT resolution estimator, (q84 - q16)/2"),
        btagUParTAK4B = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll'):-1",float,precision=10,doc="UnifiedParT b vs. udscg"),
        btagUParTAK4CvL = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL'):-1",float,precision=10,doc="UnifiedParT c vs. udsg"),
        btagUParTAK4CvB = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB'):-1",float,precision=10,doc="UnifiedParT c vs. b"),
        btagUParTAK4CvNotB = Var("?((bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probb')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:problepb')))>0?((bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probc'))/(1.-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probb')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probbb')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:problepb'))):-1",float,precision=10,doc="UnifiedParT c vs. not b"),
        btagUParTAK4QvG = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG'):-1",float,precision=10,doc="UnifiedParT q (udsbc) vs. g"),
        btagUParTAK4TauVJet = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:TauVsJet')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:TauVsJet'):-1",float,precision=10,doc="UnifiedParT tau vs. jet"),
        UParTAK4RegPtRawCorr = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr'):-1",float,precision=10,doc="UnifiedParT universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT"),
        UParTAK4RegPtRawCorrNeutrino = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu'):-1",float,precision=10,doc="UnifiedParT universal flavor-aware pT regression neutrino correction, relative to visible. To apply full regression, multiply raw jet pT by both UParTAK4RegPtRawCorr and UParTAK4RegPtRawCorrNeutrino."),
        UParTAK4RegPtRawRes = Var("?(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow'))>0?0.5*(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow')):-1",float,precision=10,doc="UnifiedParT universal flavor-aware jet pT resolution estimator, (q84 - q16)/2"),
        jetId = Var("userInt('tightId')*2+4*userInt('tightIdLepVeto')", "uint8",doc="Jet ID flag: bit2 is tight, bit3 is tightLepVeto"),
        hfsigmaEtaEta = Var("userFloat('hfsigmaEtaEta')",float,doc="sigmaEtaEta for HF jets (noise discriminating variable)",precision=10),
        hfsigmaPhiPhi = Var("userFloat('hfsigmaPhiPhi')",float,doc="sigmaPhiPhi for HF jets (noise discriminating variable)",precision=10),
        hfcentralEtaStripSize = Var("userInt('hfcentralEtaStripSize')", int, doc="eta size of the central tower strip in HF (noise discriminating variable)"),
        hfadjacentEtaStripsSize = Var("userInt('hfadjacentEtaStripsSize')", int, doc="eta size of the strips next to the central tower strip in HF (noise discriminating variable)"),
        nConstituents = Var("numberOfDaughters()","uint8",doc="Number of particles in the jet"),
        chMultiplicity = Var("chargedMultiplicity()","uint8",doc="(Puppi-weighted) Number of charged particles in the jet"),
        neMultiplicity = Var("neutralMultiplicity()","uint8",doc="(Puppi-weighted) Number of neutral particles in the jet"),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        chHEF = Var("chargedHadronEnergyFraction()", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergyFraction()", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("chargedEmEnergyFraction()", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("neutralEmEnergyFraction()", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        hfHEF = Var("HFHadronEnergyFraction()",float,doc="hadronic Energy Fraction in HF",precision= 6),
        hfEmEF = Var("HFEMEnergyFraction()",float,doc="electromagnetic Energy Fraction in HF",precision= 6),
        muEF = Var("muonEnergyFraction()", float, doc="muon Energy Fraction", precision= 6),
    )
)

run2_nanoAOD_ANY.toModify(
    jetPuppiTable.variables,
    btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
    btagDeepB = Var("?(bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'))>=0?bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'):-1",float,doc="DeepCSV b+bb tag discriminator",precision=10),
    btagDeepCvL = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probudsg')):-1", float,doc="DeepCSV c vs udsg discriminator",precision=10),
    btagDeepCvB = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')):-1",float,doc="DeepCSV c vs b+bb discriminator",precision=10)
)

(run3_nanoAOD_122 | run3_nanoAOD_124).toModify(
    jetPuppiTable.variables,
    # New ParticleNet trainings are not available in MiniAOD until Run3 13X
    btagPNetB = None,
    btagPNetCvL = None,
    btagPNetCvB = None,
    btagPNetQvG = None,
    btagPNetTauVJet = None,
    PNetRegPtRawCorr = None,
    PNetRegPtRawCorrNeutrino = None,
    PNetRegPtRawRes = None,
    # Remove for V11 and earlier Run3 versions
    chMultiplicity = None,
    neMultiplicity = None,
    hfHEF = None,
    hfEmEF = None
)

#jets are not as precise as muons
jetPuppiTable.variables.pt.precision=10

##############################################################
## DeepInfoAK4:Start
## - To be used in nanoAOD_customizeCommon() in nano_cff.py
###############################################################
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
def nanoAOD_addDeepInfoAK4(process,addParticleNet,addRobustParTAK4=False,addUnifiedParTAK4=False):
    _btagDiscriminators=[]
    if addParticleNet:
        print("Updating process to run ParticleNetAK4")
        from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll as pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
        from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll as pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
        _btagDiscriminators += pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
        _btagDiscriminators += pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
    if addRobustParTAK4:
        print("Updating process to run RobustParTAK4")
        from RecoBTag.ONNXRuntime.pfParticleTransformerAK4_cff import _pfParticleTransformerAK4JetTagsAll as pfParticleTransformerAK4JetTagsAll
        _btagDiscriminators += pfParticleTransformerAK4JetTagsAll
    if addUnifiedParTAK4:
        print("Updating process to run UnifiedParTAK4")
        from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4_cff import _pfUnifiedParticleTransformerAK4JetTagsAll as pfUnifiedParticleTransformerAK4JetTagsAll
        _btagDiscriminators += pfUnifiedParticleTransformerAK4JetTagsAll
    if len(_btagDiscriminators)==0: return process
    print("Will recalculate the following discriminators: "+", ".join(_btagDiscriminators))
    updateJetCollection(
        process,
        jetSource = cms.InputTag('slimmedJetsPuppi'),
        jetCorrections = ('AK4PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
        btagDiscriminators = _btagDiscriminators,
        postfix = 'PuppiWithDeepInfo',
    )
    process.load("Configuration.StandardSequences.MagneticField_cff")
    process.jetPuppiCorrFactorsNano.src = "selectedUpdatedPatJetsPuppiWithDeepInfo"
    process.updatedJetsPuppi.jetSource = "selectedUpdatedPatJetsPuppiWithDeepInfo"
    return process

nanoAOD_addDeepInfoAK4_switch = cms.PSet(
    nanoAOD_addParticleNet_switch = cms.untracked.bool(False),
    nanoAOD_addRobustParTAK4Tag_switch = cms.untracked.bool(False),
    nanoAOD_addUnifiedParTAK4Tag_switch = cms.untracked.bool(False)
)

################################################
## DeepInfoAK4:End
#################################################

################################################################################
# JETS FOR MET type1
################################################################################
basicJetsPuppiForMetForT1METNano = cms.EDProducer("PATJetCleanerForType1MET",
    src = updatedJetsPuppiWithUserData.src,
    jetCorrEtaMax = cms.double(9.9),
    jetCorrLabel = cms.InputTag("L3Absolute"),
    jetCorrLabelRes = cms.InputTag("L2L3Residual"),
    offsetCorrLabel = cms.InputTag("L1FastJet"),
    skipEM = cms.bool(False),
    skipEMfractionThreshold = cms.double(0.9),
    skipMuonSelection = cms.string('isGlobalMuon | isStandAloneMuon'),
    skipMuons = cms.bool(True),
    type1JetPtThreshold = cms.double(0.0),
    calcMuonSubtrRawPtAsValueMap = cms.bool(True)
)

updatedJetsPuppiWithUserData.userFloats.muonSubtrRawPt = cms.InputTag("basicJetsPuppiForMetForT1METNano:MuonSubtrRawPt")

corrT1METJetPuppiTable = simplePATJetFlatTableProducer.clone(
    src = finalJetsPuppi.src,
    cut = cms.string("pt<15 && abs(eta)<9.9"),
    name = cms.string("CorrT1METJet"),
    doc  = cms.string("Additional low-pt ak4 Puppi jets for Type-1 MET re-correction"),
    variables = cms.PSet(
        rawPt = Var("pt()*jecFactor('Uncorrected')",float,precision=10),
        eta  = Var("eta",  float,precision=12),
        phi = Var("phi", float, precision=12),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        EmEF = Var("chargedEmEnergyFraction()+neutralEmEnergyFraction()", float, doc="charged+neutral Electromagnetic Energy Fraction", precision=6),
    )
)

corrT1METJetPuppiTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)
jetPuppiTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)

jetPuppiForMETTask =  cms.Task(basicJetsPuppiForMetForT1METNano,corrT1METJetPuppiTable)

#before cross linking
jetPuppiUserDataTask = cms.Task(tightJetPuppiId, tightJetPuppiIdLepVeto, hfJetPuppiShowerShapeforNanoAOD)

#before cross linking
jetPuppiTask = cms.Task(jetPuppiCorrFactorsNano,updatedJetsPuppi,jetPuppiUserDataTask,updatedJetsPuppiWithUserData,finalJetsPuppi)
jetPuppiTablesTask = cms.Task(jetPuppiTable)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
from PhysicsTools.NanoAOD.common_cff import ExtVar

def nanoAOD_refineFastSim_puppiJet(process):
    process.puppiJetRefineNN = cms.EDProducer(
        "JetBaseMVAValueMapProducer",
        backend             = cms.string("ONNX"),
        batch_eval          = cms.bool(True),
        disableONNXGraphOpt = cms.bool(True),
        src                 = cms.InputTag("updatedJetsPuppi"),  # <<< HERE
        weightFile          = cms.FileInPath("model_refinement_regression_20250320_dynamic.onnx"),
        name                = cms.string("puppiJetRefineNN"),
        variables = cms.VPSet(
            cms.PSet(name=cms.string("GenJet_pt"),            expr=cms.string("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().pt():pt")),
            cms.PSet(name=cms.string("GenJet_eta"),           expr=cms.string("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().eta():eta")),
            cms.PSet(name=cms.string("Jet_hadronFlavour"),    expr=cms.string("hadronFlavour()")),
            cms.PSet(name=cms.string("Jet_pt"),               expr=cms.string("pt()")),
            cms.PSet(name=cms.string("Jet_btagDeepFlavB"),    expr=cms.string("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')")),
            cms.PSet(name=cms.string("Jet_btagDeepFlavCvB"),  expr=cms.string("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1")),
            cms.PSet(name=cms.string("Jet_btagDeepFlavCvL"),  expr=cms.string("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1")),
            cms.PSet(name=cms.string("Jet_btagDeepFlavQG"),   expr=cms.string("?(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1")),
            cms.PSet(name=cms.string("Jet_btagUParTAK4B"),    expr=cms.string("bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll')")),
            cms.PSet(name=cms.string("Jet_btagUParTAK4CvB"),  expr=cms.string("?(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB')>0)?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB'):-1")),
            cms.PSet(name=cms.string("Jet_btagUParTAK4CvL"),  expr=cms.string("?(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL')>0)?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL'):-1")),
            cms.PSet(name=cms.string("Jet_btagUParTAK4QvG"),  expr=cms.string("?(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG')>0)?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG'):-1")),
        ),
        inputTensorName  = cms.string("input"),
        outputTensorName = cms.string("output"),
        outputNames      = cms.vstring([
            "ptrefined","btagDeepFlavBrefined","btagDeepFlavCvBrefined","btagDeepFlavCvLrefined",
            "btagDeepFlavQGrefined","btagUParTAK4Brefined","btagUParTAK4CvBrefined",
            "btagUParTAK4CvLrefined","btagUParTAK4QvGrefined"
        ]),
        outputFormulas   = cms.vstring([f"at({i})" for i in range(9)]),
    )

    fastSim.toModify(process.jetPuppiTask, process.jetPuppiTask.add(process.puppiJetRefineNN))

    # 1) now stuff the 9 refined scalars into the existing PATJetUserDataEmbedder
    fastSim.toModify(process.updatedJetsPuppiWithUserData.userFloats,
        ptRefined              = cms.InputTag("puppiJetRefineNN","ptrefined"),
        btagDeepFlavBrefined   = cms.InputTag("puppiJetRefineNN","btagDeepFlavBrefined"),
        btagDeepFlavCvBrefined = cms.InputTag("puppiJetRefineNN","btagDeepFlavCvBrefined"),
        btagDeepFlavCvLrefined = cms.InputTag("puppiJetRefineNN","btagDeepFlavCvLrefined"),
        btagDeepFlavQGrefined  = cms.InputTag("puppiJetRefineNN","btagDeepFlavQGrefined"),
        btagUParTAK4Brefined   = cms.InputTag("puppiJetRefineNN","btagUParTAK4Brefined"),
        btagUParTAK4CvBrefined = cms.InputTag("puppiJetRefineNN","btagUParTAK4CvBrefined"),
        btagUParTAK4CvLrefined = cms.InputTag("puppiJetRefineNN","btagUParTAK4CvLrefined"),
        btagUParTAK4QvGrefined = cms.InputTag("puppiJetRefineNN","btagUParTAK4QvGrefined"),
    )

    # 2) backup all 9 originals in the NanoAOD table as *_unrefined
    fastSim.toModify(process.jetPuppiTable.variables,
        pt_unrefined               = process.jetPuppiTable.variables.pt.clone(),
        btagDeepFlavB_unrefined    = process.jetPuppiTable.variables.btagDeepFlavB.clone(),
        btagDeepFlavCvB_unrefined  = process.jetPuppiTable.variables.btagDeepFlavCvB.clone(),
        btagDeepFlavCvL_unrefined  = process.jetPuppiTable.variables.btagDeepFlavCvL.clone(),
        btagDeepFlavQG_unrefined   = process.jetPuppiTable.variables.btagDeepFlavQG.clone(),
        btagUParTAK4B_unrefined    = process.jetPuppiTable.variables.btagUParTAK4B.clone(),
        btagUParTAK4CvB_unrefined  = process.jetPuppiTable.variables.btagUParTAK4CvB.clone(),
        btagUParTAK4CvL_unrefined  = process.jetPuppiTable.variables.btagUParTAK4CvL.clone(),
        btagUParTAK4QvG_unrefined  = process.jetPuppiTable.variables.btagUParTAK4QvG.clone(),
    )
    # drop them so we can redefine
    fastSim.toModify(process.jetPuppiTable.variables,
        pt=None, btagDeepFlavB=None, btagDeepFlavCvB=None,
        btagDeepFlavCvL=None, btagDeepFlavQG=None,
        btagUParTAK4B=None, btagUParTAK4CvB=None,
        btagUParTAK4CvL=None, btagUParTAK4QvG=None,
    )

    # 3) re‑define each branch with a FastSim‑only mask ternary
    mask = "(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll')>0)"

    fastSim.toModify(process.jetPuppiTable.variables,
        pt              = Var(f"?{mask}?userFloat('ptRefined'):pt()",              float, precision=10, doc="refined pT or original"),
        btagDeepFlavB   = Var(f"?{mask}?userFloat('btagDeepFlavBrefined'):(bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))", float, precision=10),
        btagDeepFlavCvB = Var(f"?{mask}?userFloat('btagDeepFlavCvBrefined'):(?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1)", float, precision=10),
        btagDeepFlavCvL = Var(f"?{mask}?userFloat('btagDeepFlavCvLrefined'):(?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1)", float, precision=10),
        btagDeepFlavQG  = Var(f"?{mask}?userFloat('btagDeepFlavQGrefined'):(?(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1)", float, precision=10),
        btagUParTAK4B   = Var(f"?{mask}?userFloat('btagUParTAK4Brefined'):(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll'))", float, precision=10),
        btagUParTAK4CvB = Var(f"?{mask}?userFloat('btagUParTAK4CvBrefined'):(?(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB')>0)?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB'):-1)", float, precision=10),
        btagUParTAK4CvL = Var(f"?{mask}?userFloat('btagUParTAK4CvLrefined'):(?(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL')>0)?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL'):-1)", float, precision=10),
        btagUParTAK4QvG = Var(f"?{mask}?userFloat('btagUParTAK4QvGrefined'):(?(bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG')>0)?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG'):-1)", float, precision=10),
    )

    return process

nanoAOD_refineFastSim_puppiJet = nanoAOD_refineFastSim_puppiJet
