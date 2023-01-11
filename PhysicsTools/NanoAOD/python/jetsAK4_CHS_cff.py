import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

##################### User floats producers, selectors ##########################

from  PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
# Note: Safe to always add 'L2L3Residual' as MC contains dummy L2L3Residual corrections (always set to 1)
#      (cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#CMSSW_7_6_4_and_above )
jetCorrFactorsNano = patJetCorrFactors.clone(src='slimmedJets',
    levels = cms.vstring('L1FastJet',
        'L2Relative',
        'L3Absolute',
        'L2L3Residual'),
    primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
)

from  PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import *
updatedJets = updatedPatJets.clone(
    addBTagInfo=False,
    jetSource='slimmedJets',
    jetCorrFactorsSource=cms.VInputTag(cms.InputTag("jetCorrFactorsNano") ),
)

#
# JetID
#
looseJetId = cms.EDProducer("PatJetIDValueMapProducer",
    filterParams=cms.PSet(
        version = cms.string('WINTER16'),
        quality = cms.string('LOOSE'),
    ),
    src = cms.InputTag("updatedJets")
)
tightJetId = cms.EDProducer("PatJetIDValueMapProducer",
    filterParams=cms.PSet(
        version = cms.string('RUN2ULCHS'),
        quality = cms.string('TIGHT'),
    ),
    src = cms.InputTag("updatedJets")
)
tightJetIdLepVeto = cms.EDProducer("PatJetIDValueMapProducer",
    filterParams=cms.PSet(
        version = cms.string('RUN2ULCHS'),
        quality = cms.string('TIGHTLEPVETO'),
    ),
    src = cms.InputTag("updatedJets")
)
run2_jme_2016.toModify(
    tightJetId.filterParams, version = "RUN2UL16CHS"
).toModify(
    tightJetIdLepVeto.filterParams, version = "RUN2UL16CHS"
)

bJetVars = cms.EDProducer("JetRegressionVarProducer",
    pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    src = cms.InputTag("updatedJets"),
    svsrc = cms.InputTag("slimmedSecondaryVertices"),
)

jercVars = cms.EDProducer("BetaStarPackedCandidateVarProducer",
    srcJet = cms.InputTag("updatedJets"),
    srcPF = cms.InputTag("packedPFCandidates"),
    maxDR = cms.double(0.4)
)

updatedJetsWithUserData = cms.EDProducer("PATJetUserDataEmbedder",
    src = cms.InputTag("updatedJets"),
    userFloats = cms.PSet(
        leadTrackPt = cms.InputTag("bJetVars:leadTrackPt"),
        leptonPtRelv0 = cms.InputTag("bJetVars:leptonPtRelv0"),
        leptonPtRelInvv0 = cms.InputTag("bJetVars:leptonPtRelInvv0"),
        leptonDeltaR = cms.InputTag("bJetVars:leptonDeltaR"),
        vtxPt = cms.InputTag("bJetVars:vtxPt"),
        vtxMass = cms.InputTag("bJetVars:vtxMass"),
        vtx3dL = cms.InputTag("bJetVars:vtx3dL"),
        vtx3deL = cms.InputTag("bJetVars:vtx3deL"),
        ptD = cms.InputTag("bJetVars:ptD"),
        qgl = cms.InputTag('qgtagger:qgLikelihood'),
        puIdNanoDisc = cms.InputTag('pileupJetIdNano:fullDiscriminant'),
        chFPV0EF = cms.InputTag("jercVars:chargedFromPV0EnergyFraction"),
    ),
    userInts = cms.PSet(
        tightId = cms.InputTag("tightJetId"),
        tightIdLepVeto = cms.InputTag("tightJetIdLepVeto"),
        vtxNtrk = cms.InputTag("bJetVars:vtxNtrk"),
        leptonPdgId = cms.InputTag("bJetVars:leptonPdgId"),
        puIdNanoId = cms.InputTag('pileupJetIdNano:fullId'),
    ),
)


finalJets = cms.EDFilter("PATJetRefSelector",
    src = cms.InputTag("updatedJetsWithUserData"),
    cut = cms.string("pt > 15")
)


##################### Tables for final output and docs ##########################


jetTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("linkedObjects","jets"),
    name = cms.string("Jet"),
    doc  = cms.string("slimmedJets, i.e. ak4 PFJets CHS with JECs applied, after basic selection (" + finalJets.cut.value()+")"),
    externalVariables = cms.PSet(
        bRegCorr = ExtVar(cms.InputTag("bjetNN:corr"),float, doc="pt correction for b-jet energy regression",precision=10),
        bRegRes = ExtVar(cms.InputTag("bjetNN:res"),float, doc="res on pt corrected with b-jet regression",precision=6),
        cRegCorr = ExtVar(cms.InputTag("cjetNN:corr"),float, doc="pt correction for c-jet energy regression",precision=10),
        cRegRes = ExtVar(cms.InputTag("cjetNN:res"),float, doc="res on pt corrected with c-jet regression",precision=6),
    ),
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
        btagDeepB = Var("?(bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'))>=0?bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'):-1",float,doc="DeepCSV b+bb tag discriminator",precision=10),
        btagDeepFlavB = Var("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')",float,doc="DeepJet b+bb+lepb tag discriminator",precision=10),
        btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
        btagDeepCvL = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probudsg')):-1", float,doc="DeepCSV c vs udsg discriminator",precision=10),
        btagDeepCvB = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')):-1",float,doc="DeepCSV c vs b+bb discriminator",precision=10),
        btagDeepFlavCvL = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1",float,doc="DeepJet c vs uds+g discriminator",precision=10),
        btagDeepFlavCvB = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1",float,doc="DeepJet c vs b+bb+lepb discriminator",precision=10),
        btagDeepFlavQG = Var("?(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1",float,doc="DeepJet g vs uds discriminator",precision=10),
        puIdDisc = Var("userFloat('puIdNanoDisc')", float,doc="Pileup ID discriminant with 106X (2018) training",precision=10),
        puId = Var("userInt('puIdNanoId')", "uint8", doc="Pileup ID flags with 106X (2018) training"),
        jetId = Var("userInt('tightId')*2+4*userInt('tightIdLepVeto')", "uint8", doc="Jet ID flags bit1 is loose (always false in 2017 since it does not exist), bit2 is tight, bit3 is tightLepVeto"),
        qgl = Var("?userFloat('qgl')>0?userFloat('qgl'):-1",float,doc="Quark vs Gluon likelihood discriminator",precision=10),
        hfsigmaEtaEta = Var("userFloat('hfJetShowerShape:sigmaEtaEta')",float,doc="sigmaEtaEta for HF jets (noise discriminating variable)",precision=10),
        hfsigmaPhiPhi = Var("userFloat('hfJetShowerShape:sigmaPhiPhi')",float,doc="sigmaPhiPhi for HF jets (noise discriminating variable)",precision=10),
        hfcentralEtaStripSize = Var("userInt('hfJetShowerShape:centralEtaStripSize')", int, doc="eta size of the central tower strip in HF (noise discriminating variable) "),
        hfadjacentEtaStripsSize = Var("userInt('hfJetShowerShape:adjacentEtaStripsSize')", int, doc="eta size of the strips next to the central tower strip in HF (noise discriminating variable) "),
        nConstituents = Var("numberOfDaughters()","uint8",doc="Number of particles in the jet"),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        chHEF = Var("chargedHadronEnergyFraction()", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergyFraction()", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("chargedEmEnergyFraction()", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("neutralEmEnergyFraction()", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        muEF = Var("muonEnergyFraction()", float, doc="muon Energy Fraction", precision= 6),
        chFPV0EF = Var("userFloat('chFPV0EF')", float, doc="charged fromPV==0 Energy Fraction (energy excluded from CHS jets). Previously called betastar.", precision= 6),
    )
)

#jets are not as precise as muons
jetTable.variables.pt.precision=10

### Era dependent customization
(run2_jme_2016 & ~tracker_apv_vfp30_2016 ).toModify(
    jetTable.variables.puIdDisc, doc="Pileup ID discriminant with 106X (2016) training"
).toModify(
    jetTable.variables.puId, doc="Pileup ID flags with 106X (2016) training"
)
(run2_jme_2016 & tracker_apv_vfp30_2016 ).toModify(
    jetTable.variables.puIdDisc, doc="Pileup ID discriminant with 106X (2016APV) training"
).toModify(
    jetTable.variables.puId,  doc="Pileup ID flags with 106X (2016APV) training"
)
run2_jme_2017.toModify(
    jetTable.variables.puIdDisc, doc="Pileup ID discriminant with 106X (2017) training"
).toModify(
    jetTable.variables.puId, doc="Pileup ID flags with 106X (2017) training"
)

bjetNN = cms.EDProducer("BJetEnergyRegressionMVA",
    backend = cms.string("ONNX"),
    batch_eval = cms.bool(True),
    src = cms.InputTag("linkedObjects","jets"),
    pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    svsrc = cms.InputTag("slimmedSecondaryVertices"),
    rhosrc = cms.InputTag("fixedGridRhoFastjetAll"),

    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/breg_training_2018.onnx"),
    name = cms.string("JetRegNN"),
    isClassifier = cms.bool(False),
    variablesOrder = cms.vstring(["Jet_pt","Jet_eta","rho","Jet_mt","Jet_leadTrackPt","Jet_leptonPtRel","Jet_leptonDeltaR","Jet_neHEF",
                                  "Jet_neEmEF","Jet_vtxPt","Jet_vtxMass","Jet_vtx3dL","Jet_vtxNtrk","Jet_vtx3deL",
                                  "Jet_numDaughters_pt03","Jet_energyRing_dR0_em_Jet_rawEnergy","Jet_energyRing_dR1_em_Jet_rawEnergy",
                                  "Jet_energyRing_dR2_em_Jet_rawEnergy","Jet_energyRing_dR3_em_Jet_rawEnergy","Jet_energyRing_dR4_em_Jet_rawEnergy",
                                  "Jet_energyRing_dR0_neut_Jet_rawEnergy","Jet_energyRing_dR1_neut_Jet_rawEnergy","Jet_energyRing_dR2_neut_Jet_rawEnergy",
                                  "Jet_energyRing_dR3_neut_Jet_rawEnergy","Jet_energyRing_dR4_neut_Jet_rawEnergy","Jet_energyRing_dR0_ch_Jet_rawEnergy",
                                  "Jet_energyRing_dR1_ch_Jet_rawEnergy","Jet_energyRing_dR2_ch_Jet_rawEnergy","Jet_energyRing_dR3_ch_Jet_rawEnergy",
                                  "Jet_energyRing_dR4_ch_Jet_rawEnergy","Jet_energyRing_dR0_mu_Jet_rawEnergy","Jet_energyRing_dR1_mu_Jet_rawEnergy",
                                  "Jet_energyRing_dR2_mu_Jet_rawEnergy","Jet_energyRing_dR3_mu_Jet_rawEnergy","Jet_energyRing_dR4_mu_Jet_rawEnergy",
                                  "Jet_chHEF","Jet_chEmEF","Jet_leptonPtRelInv","isEle","isMu","isOther","Jet_mass","Jet_ptd"]),
    variables = cms.PSet(
    Jet_pt = cms.string("pt*jecFactor('Uncorrected')"),
    Jet_mt = cms.string("mt*jecFactor('Uncorrected')"),
    Jet_eta = cms.string("eta"),
    Jet_mass = cms.string("mass*jecFactor('Uncorrected')"),
    Jet_ptd = cms.string("userFloat('ptD')"),
    Jet_leadTrackPt = cms.string("userFloat('leadTrackPt')"),
    Jet_vtxNtrk = cms.string("userInt('vtxNtrk')"),
    Jet_vtxMass = cms.string("userFloat('vtxMass')"),
    Jet_vtx3dL = cms.string("userFloat('vtx3dL')"),
    Jet_vtx3deL = cms.string("userFloat('vtx3deL')"),
    Jet_vtxPt = cms.string("userFloat('vtxPt')"),
    Jet_leptonPtRel = cms.string("userFloat('leptonPtRelv0')"),
    Jet_leptonPtRelInv = cms.string("userFloat('leptonPtRelInvv0')*jecFactor('Uncorrected')"),
    Jet_leptonDeltaR = cms.string("userFloat('leptonDeltaR')"),
    Jet_neHEF = cms.string("neutralHadronEnergyFraction()"),
    Jet_neEmEF = cms.string("neutralEmEnergyFraction()"),
    Jet_chHEF = cms.string("chargedHadronEnergyFraction()"),
    Jet_chEmEF = cms.string("chargedEmEnergyFraction()"),
    isMu = cms.string("?abs(userInt('leptonPdgId'))==13?1:0"),
    isEle = cms.string("?abs(userInt('leptonPdgId'))==11?1:0"),
    isOther = cms.string("?userInt('leptonPdgId')==0?1:0"),
    ),
     inputTensorName = cms.string("ffwd_inp:0"),
     outputTensorName = cms.string("ffwd_out/BiasAdd:0"),
     outputNames = cms.vstring(["corr","res"]),
     outputFormulas = cms.vstring(["at(0)*0.27912887930870056+1.0545977354049683","0.5*(at(2)-at(1))*0.27912887930870056"]),
)

cjetNN = cms.EDProducer("BJetEnergyRegressionMVA",
    backend = cms.string("ONNX"),
    batch_eval = cms.bool(True),

    src = cms.InputTag("linkedObjects","jets"),
    pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    svsrc = cms.InputTag("slimmedSecondaryVertices"),
    rhosrc = cms.InputTag("fixedGridRhoFastjetAll"),

    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/creg_training_2018.onnx"),
    name = cms.string("JetRegNN"),
    isClassifier = cms.bool(False),
    variablesOrder = cms.vstring(["Jet_pt","Jet_eta","rho","Jet_mt","Jet_leadTrackPt","Jet_leptonPtRel","Jet_leptonDeltaR",
                                  "Jet_neHEF","Jet_neEmEF","Jet_vtxPt","Jet_vtxMass","Jet_vtx3dL","Jet_vtxNtrk","Jet_vtx3deL",
                                  "Jet_numDaughters_pt03","Jet_chEmEF","Jet_chHEF", "Jet_ptd","Jet_mass",
                                  "Jet_energyRing_dR0_em_Jet_rawEnergy","Jet_energyRing_dR1_em_Jet_rawEnergy",
                                  "Jet_energyRing_dR2_em_Jet_rawEnergy","Jet_energyRing_dR3_em_Jet_rawEnergy","Jet_energyRing_dR4_em_Jet_rawEnergy",
                                  "Jet_energyRing_dR0_neut_Jet_rawEnergy","Jet_energyRing_dR1_neut_Jet_rawEnergy","Jet_energyRing_dR2_neut_Jet_rawEnergy",
                                  "Jet_energyRing_dR3_neut_Jet_rawEnergy","Jet_energyRing_dR4_neut_Jet_rawEnergy","Jet_energyRing_dR0_ch_Jet_rawEnergy",
                                  "Jet_energyRing_dR1_ch_Jet_rawEnergy","Jet_energyRing_dR2_ch_Jet_rawEnergy","Jet_energyRing_dR3_ch_Jet_rawEnergy",
                                  "Jet_energyRing_dR4_ch_Jet_rawEnergy","Jet_energyRing_dR0_mu_Jet_rawEnergy","Jet_energyRing_dR1_mu_Jet_rawEnergy",
                                  "Jet_energyRing_dR2_mu_Jet_rawEnergy","Jet_energyRing_dR3_mu_Jet_rawEnergy","Jet_energyRing_dR4_mu_Jet_rawEnergy"]),
    variables = cms.PSet(
    Jet_pt = cms.string("pt*jecFactor('Uncorrected')"),
    Jet_mt = cms.string("mt*jecFactor('Uncorrected')"),
    Jet_eta = cms.string("eta"),
    Jet_mass = cms.string("mass*jecFactor('Uncorrected')"),
    Jet_ptd = cms.string("userFloat('ptD')"),
    Jet_leadTrackPt = cms.string("userFloat('leadTrackPt')"),
    Jet_vtxNtrk = cms.string("userInt('vtxNtrk')"),
    Jet_vtxMass = cms.string("userFloat('vtxMass')"),
    Jet_vtx3dL = cms.string("userFloat('vtx3dL')"),
    Jet_vtx3deL = cms.string("userFloat('vtx3deL')"),
    Jet_vtxPt = cms.string("userFloat('vtxPt')"),
    Jet_leptonPtRel = cms.string("userFloat('leptonPtRelv0')"),
    Jet_leptonPtRelInv = cms.string("userFloat('leptonPtRelInvv0')*jecFactor('Uncorrected')"),
    Jet_leptonDeltaR = cms.string("userFloat('leptonDeltaR')"),
    Jet_neHEF = cms.string("neutralHadronEnergyFraction()"),
    Jet_neEmEF = cms.string("neutralEmEnergyFraction()"),
    Jet_chHEF = cms.string("chargedHadronEnergyFraction()"),
    Jet_chEmEF = cms.string("chargedEmEnergyFraction()"),
    isMu = cms.string("?abs(userInt('leptonPdgId'))==13?1:0"),
    isEle = cms.string("?abs(userInt('leptonPdgId'))==11?1:0"),
    isOther = cms.string("?userInt('leptonPdgId')==0?1:0"),
    ),
    inputTensorName = cms.string("ffwd_inp:0"),
    outputTensorName = cms.string("ffwd_out/BiasAdd:0"),
    outputNames = cms.vstring(["corr","res"]),
    outputFormulas = cms.vstring(["at(0)*0.24325256049633026+0.993854820728302","0.5*(at(2)-at(1))*0.24325256049633026"]),
)


run2_jme_2016.toModify(
    bjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/breg_training_2016.onnx")
).toModify(
    bjetNN,outputFormulas = cms.vstring(["at(0)*0.31976690888404846+1.047176718711853","0.5*(at(2)-at(1))*0.31976690888404846"])
).toModify(
    cjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/creg_training_2016.onnx")
).toModify(
    cjetNN, outputFormulas = cms.vstring(["at(0)*0.28862622380256653+0.9908722639083862","0.5*(at(2)-at(1))*0.28862622380256653"])
)

run2_jme_2017.toModify(
    bjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/breg_training_2017.onnx")
).toModify(
    bjetNN,outputFormulas = cms.vstring(["at(0)*0.28225210309028625+1.055067777633667","0.5*(at(2)-at(1))*0.28225210309028625"])
).toModify(
    cjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/creg_training_2017.onnx")
).toModify(
    cjetNN, outputFormulas = cms.vstring(["at(0)*0.24718524515628815+0.9927206635475159","0.5*(at(2)-at(1))*0.24718524515628815"])
)


#
# Quark-Gluon Likelihood (QGL)
#
from RecoJets.JetProducers.QGTagger_cfi import  QGTagger
qgtagger=QGTagger.clone(srcJets="updatedJets",srcVertexCollection="offlineSlimmedPrimaryVertices")

#
# PileUp ID
#
from RecoJets.JetProducers.PileupJetID_cfi import pileupJetId, _chsalgos_94x, _chsalgos_102x, _chsalgos_106X_UL16, _chsalgos_106X_UL16APV, _chsalgos_106X_UL17, _chsalgos_106X_UL18
pileupJetIdNano=pileupJetId.clone(jets="updatedJets",algos = cms.VPSet(_chsalgos_106X_UL18),inputIsCorrected=True,applyJec=False,vertexes="offlineSlimmedPrimaryVertices")
run2_jme_2017.toModify(
    pileupJetIdNano, algos = _chsalgos_106X_UL17
)
(run2_jme_2016 & ~tracker_apv_vfp30_2016 ).toModify(
    pileupJetIdNano, algos = _chsalgos_106X_UL16
)
(run2_jme_2016 & tracker_apv_vfp30_2016 ).toModify(
    pileupJetIdNano, algos = _chsalgos_106X_UL16APV
)


##############################################################
## DeepInfoAK4CHS:Start
## - To be used in nanoAOD_customizeCommon() in nano_cff.py
###############################################################
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
def nanoAOD_addDeepInfoAK4CHS(process,addDeepBTag,addDeepFlavour):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb','pfDeepCSVJetTags:probc']
    if addDeepFlavour:
        print("Updating process to run DeepFlavour btag")
        _btagDiscriminators += ['pfDeepFlavourJetTags:probb','pfDeepFlavourJetTags:probbb','pfDeepFlavourJetTags:problepb','pfDeepFlavourJetTags:probc']
    if len(_btagDiscriminators)==0: return process
    print("Will recalculate the following discriminators: "+", ".join(_btagDiscriminators))
    updateJetCollection(
        process,
        jetSource = cms.InputTag('slimmedJets'),
        jetCorrections = ('AK4PFchs', cms.vstring(['L1FastJet', 'L2Relative', 'L3Absolute','L2L3Residual']), 'None'),
        btagDiscriminators = _btagDiscriminators,
        postfix = 'WithDeepInfo',
    )
    process.load("Configuration.StandardSequences.MagneticField_cff")
    process.jetCorrFactorsNano.src="selectedUpdatedPatJetsWithDeepInfo"
    process.updatedJets.jetSource="selectedUpdatedPatJetsWithDeepInfo"
    return process

nanoAOD_addDeepInfoAK4CHS_switch = cms.PSet(
    nanoAOD_addDeepBTag_switch = cms.untracked.bool(False),
    nanoAOD_addDeepFlavourTag_switch = cms.untracked.bool(False),
)

################################################
## DeepInfoAK4CHS:End
#################################################

################################################################################
# JETS FOR MET type1
################################################################################
basicJetsForMetForT1METNano = cms.EDProducer("PATJetCleanerForType1MET",
    src = updatedJetsWithUserData.src,
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

updatedJetsWithUserData.userFloats.muonSubtrRawPt = cms.InputTag("basicJetsForMetForT1METNano:MuonSubtrRawPt")

corrT1METJetTable = simpleCandidateFlatTableProducer.clone(
    src = finalJets.src,
    cut = cms.string("pt<15 && abs(eta)<9.9"),
    name = cms.string("CorrT1METJet"),
    doc  = cms.string("Additional low-pt ak4 CHS jets for Type-1 MET re-correction"),
    variables = cms.PSet(
        rawPt = Var("pt()*jecFactor('Uncorrected')",float,precision=10),
        eta  = Var("eta",  float,precision=12),
        phi = Var("phi", float, precision=12),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
    )
)

corrT1METJetTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)
jetTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)

jetForMETTask =  cms.Task(basicJetsForMetForT1METNano,corrT1METJetTable)

#before cross linking
jetUserDataTask = cms.Task(bJetVars,qgtagger,jercVars,tightJetId,tightJetIdLepVeto,pileupJetIdNano)

#before cross linking
jetTask = cms.Task(jetCorrFactorsNano,updatedJets,jetUserDataTask,updatedJetsWithUserData,finalJets)

#after cross linkining
jetTablesTask = cms.Task(bjetNN,cjetNN,jetTable)
