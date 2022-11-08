import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

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
        version = cms.string('RUN2ULPUPPI'),
        quality = cms.string('TIGHT'),
    ),
    src = cms.InputTag("updatedJetsPuppi")
)
tightJetPuppiIdLepVeto = cms.EDProducer("PatJetIDValueMapProducer",
    filterParams=cms.PSet(
        version = cms.string('RUN2ULPUPPI'),
        quality = cms.string('TIGHTLEPVETO'),
    ),
    src = cms.InputTag("updatedJetsPuppi")
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
jetPuppiTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","jets"),
    name = cms.string("Jet"),
    doc  = cms.string("slimmedJetsPuppi, i.e. ak4 PFJets Puppi with JECs applied, after basic selection (" + finalJetsPuppi.cut.value()+")"),
    externalVariables = cms.PSet(),
    variables = cms.PSet(P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        nMuons = Var("?hasOverlaps('muons')?overlaps('muons').size():0", int, doc="number of muons in the jet"),
        muonIdx1 = Var("?overlaps('muons').size()>0?overlaps('muons')[0].key():-1", int, doc="index of first matching muon"),
        muonIdx2 = Var("?overlaps('muons').size()>1?overlaps('muons')[1].key():-1", int, doc="index of second matching muon"),
        electronIdx1 = Var("?overlaps('electrons').size()>0?overlaps('electrons')[0].key():-1", int, doc="index of first matching electron"),
        electronIdx2 = Var("?overlaps('electrons').size()>1?overlaps('electrons')[1].key():-1", int, doc="index of second matching electron"),
        nElectrons = Var("?hasOverlaps('electrons')?overlaps('electrons').size():0", int, doc="number of electrons in the jet"),
        btagDeepB = Var("?(bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'))>=0?bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'):-1",float,doc="DeepCSV b+bb tag discriminator",precision=10),
        btagDeepFlavB = Var("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')",float,doc="DeepJet b+bb+lepb tag discriminator",precision=10),
        btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
        btagDeepCvL = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probudsg')):-1", float,doc="DeepCSV c vs udsg discriminator",precision=10),
        btagDeepCvB = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')):-1",float,doc="DeepCSV c vs b+bb discriminator",precision=10),
        btagDeepFlavCvL = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1",float,doc="DeepJet c vs uds+g discriminator",precision=10),
        btagDeepFlavCvB = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1",float,doc="DeepJet c vs b+bb+lepb discriminator",precision=10),
        btagDeepFlavQG = Var("?(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1",float,doc="DeepJet g vs uds discriminator",precision=10),
        jetId = Var("userInt('tightId')*2+4*userInt('tightIdLepVeto')",int,doc="Jet ID flag: bit2 is tight, bit3 is tightLepVeto"),
        hfsigmaEtaEta = Var("userFloat('hfsigmaEtaEta')",float,doc="sigmaEtaEta for HF jets (noise discriminating variable)",precision=10),
        hfsigmaPhiPhi = Var("userFloat('hfsigmaPhiPhi')",float,doc="sigmaPhiPhi for HF jets (noise discriminating variable)",precision=10),
        hfcentralEtaStripSize = Var("userInt('hfcentralEtaStripSize')", int, doc="eta size of the central tower strip in HF (noise discriminating variable)"),
        hfadjacentEtaStripsSize = Var("userInt('hfadjacentEtaStripsSize')", int, doc="eta size of the strips next to the central tower strip in HF (noise discriminating variable)"),
        nConstituents = Var("numberOfDaughters()","uint8",doc="Number of particles in the jet"),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        chHEF = Var("chargedHadronEnergyFraction()", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergyFraction()", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("chargedEmEnergyFraction()", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("neutralEmEnergyFraction()", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        muEF = Var("muonEnergyFraction()", float, doc="muon Energy Fraction", precision= 6),
    )
)

#jets are not as precise as muons
jetPuppiTable.variables.pt.precision=10

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

corrT1METJetPuppiTable = simpleCandidateFlatTableProducer.clone(
    src = finalJetsPuppi.src,
    cut = cms.string("pt<15 && abs(eta)<9.9"),
    name = cms.string("CorrT1METJet"),
    doc  = cms.string("Additional low-pt ak4 Puppi jets for Type-1 MET re-correction"),
    variables = cms.PSet(
        rawPt = Var("pt()*jecFactor('Uncorrected')",float,precision=10),
        eta  = Var("eta",  float,precision=12),
        phi = Var("phi", float, precision=12),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
    )
)

corrT1METJetPuppiTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)
jetPuppiTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)

jetPuppiForMETTask =  cms.Task(basicJetsPuppiForMetForT1METNano,corrT1METJetPuppiTable)

#before cross linking
jetPuppiUserDataTask = cms.Task(tightJetPuppiId, tightJetPuppiIdLepVeto, hfJetPuppiShowerShapeforNanoAOD)

#before cross linking
jetPuppiTask = cms.Task(jetPuppiCorrFactorsNano,updatedJetsPuppi,jetPuppiUserDataTask,updatedJetsPuppiWithUserData,finalJetsPuppi)

#after cross linkining
jetPuppiTablesTask = cms.Task(jetPuppiTable)
