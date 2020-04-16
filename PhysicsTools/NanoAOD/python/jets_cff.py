import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_jme_2016_cff import run2_jme_2016
from Configuration.Eras.Modifier_run2_jme_2017_cff import run2_jme_2017
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

from  PhysicsTools.NanoAOD.common_cff import *
from RecoJets.JetProducers.ak4PFJetsBetaStar_cfi import *


##################### User floats producers, selectors ##########################
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

chsForSATkJets = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string('charge()!=0 && pvAssociationQuality()>=5 && vertexRef().key()==0'))
softActivityJets = ak4PFJets.clone(src = 'chsForSATkJets', doAreaFastjet = False, jetPtMin=1) 
softActivityJets10 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("softActivityJets"), cut = cms.string('pt>10'))
softActivityJets5 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("softActivityJets"), cut = cms.string('pt>5'))
softActivityJets2 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("softActivityJets"), cut = cms.string('pt>2'))

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
jetCorrFactorsAK8 = patJetCorrFactors.clone(src='slimmedJetsAK8',
    levels = cms.vstring('L1FastJet',
        'L2Relative',
        'L3Absolute',
	'L2L3Residual'),
    payload = cms.string('AK8PFPuppi'),
    primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
)
run2_miniAOD_80XLegacy.toModify(jetCorrFactorsAK8, payload = cms.string('AK8PFchs')) # ak8PFJetsCHS in 2016 80X miniAOD

from  PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import *

updatedJets = updatedPatJets.clone(
	addBTagInfo=False,
	jetSource='slimmedJets',
	jetCorrFactorsSource=cms.VInputTag(cms.InputTag("jetCorrFactorsNano") ),
)

updatedJetsAK8 = updatedPatJets.clone(
	addBTagInfo=False,
	jetSource='slimmedJetsAK8',
	jetCorrFactorsSource=cms.VInputTag(cms.InputTag("jetCorrFactorsAK8") ),
)


looseJetId = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('WINTER16'),
			    quality = cms.string('LOOSE'),
			  ),
                          src = cms.InputTag("updatedJets")
)
tightJetId = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('SUMMER18'),
			    quality = cms.string('TIGHT'),
			  ),
                          src = cms.InputTag("updatedJets")
)
tightJetIdLepVeto = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('SUMMER18'),
			    quality = cms.string('TIGHTLEPVETO'),
			  ),
                          src = cms.InputTag("updatedJets")
)
run2_jme_2016.toModify( tightJetId.filterParams, version = "WINTER16" )
run2_jme_2016.toModify( tightJetIdLepVeto.filterParams, version = "WINTER16" )
run2_jme_2017.toModify( tightJetId.filterParams, version = "WINTER17" )
run2_jme_2017.toModify( tightJetIdLepVeto.filterParams, version = "WINTER17" )


looseJetIdAK8 = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('WINTER16'),
			    quality = cms.string('LOOSE'),
			  ),
                          src = cms.InputTag("updatedJetsAK8")
)
tightJetIdAK8 = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('SUMMER18PUPPI'),
			    quality = cms.string('TIGHT'),
			  ),
                          src = cms.InputTag("updatedJetsAK8")
)
tightJetIdLepVetoAK8 = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('SUMMER18PUPPI'),
			    quality = cms.string('TIGHTLEPVETO'),
			  ),
                          src = cms.InputTag("updatedJetsAK8")
)
run2_jme_2016.toModify( tightJetIdAK8.filterParams, version = "WINTER16" )
run2_jme_2016.toModify( tightJetIdLepVetoAK8.filterParams, version = "WINTER16" )
run2_jme_2017.toModify( tightJetIdAK8.filterParams, version = "WINTER17PUPPI" )
run2_jme_2017.toModify( tightJetIdLepVetoAK8.filterParams, version = "WINTER17PUPPI" )


bJetVars = cms.EDProducer("JetRegressionVarProducer",
    pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    src = cms.InputTag("updatedJets"),
    svsrc = cms.InputTag("slimmedSecondaryVertices"),
    gpsrc = cms.InputTag("prunedGenParticles"),
    #musrc = cms.InputTag("slimmedMuons"),
    #elesrc = cms.InputTag("slimmedElectrons")
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
         leptonPtRel = cms.InputTag("bJetVars:leptonPtRel"),
         leptonPtRatio = cms.InputTag("bJetVars:leptonPtRatio"),
         leptonPtRelInv = cms.InputTag("bJetVars:leptonPtRelInv"),
         leptonPtRelv0 = cms.InputTag("bJetVars:leptonPtRelv0"),
         leptonPtRatiov0 = cms.InputTag("bJetVars:leptonPtRatiov0"),
         leptonPtRelInvv0 = cms.InputTag("bJetVars:leptonPtRelInvv0"),
         leptonDeltaR = cms.InputTag("bJetVars:leptonDeltaR"),
         leptonPt = cms.InputTag("bJetVars:leptonPt"),
         vtxPt = cms.InputTag("bJetVars:vtxPt"),
         vtxMass = cms.InputTag("bJetVars:vtxMass"),
         vtx3dL = cms.InputTag("bJetVars:vtx3dL"),
         vtx3deL = cms.InputTag("bJetVars:vtx3deL"),
         ptD = cms.InputTag("bJetVars:ptD"),
         genPtwNu = cms.InputTag("bJetVars:genPtwNu"),
         qgl = cms.InputTag('qgtagger:qgLikelihood'),
         puId94XDisc = cms.InputTag('pileupJetId94X:fullDiscriminant'),
         puId102XDisc = cms.InputTag('pileupJetId102X:fullDiscriminant'),
         chFPV0EF = cms.InputTag("jercVars:chargedFromPV0EnergyFraction"),
         chFPV1EF = cms.InputTag("jercVars:chargedFromPV1EnergyFraction"),
         chFPV2EF = cms.InputTag("jercVars:chargedFromPV2EnergyFraction"),
         chFPV3EF = cms.InputTag("jercVars:chargedFromPV3EnergyFraction"),         
         ),
     userInts = cms.PSet(
        tightId = cms.InputTag("tightJetId"),
        tightIdLepVeto = cms.InputTag("tightJetIdLepVeto"),
        vtxNtrk = cms.InputTag("bJetVars:vtxNtrk"),
        leptonPdgId = cms.InputTag("bJetVars:leptonPdgId"),
     ),
)
run2_jme_2016.toModify(updatedJetsWithUserData.userInts,
    looseId = cms.InputTag("looseJetId"),
)

updatedJetsAK8WithUserData = cms.EDProducer("PATJetUserDataEmbedder",
     src = cms.InputTag("updatedJetsAK8"),
      userInts = cms.PSet(
        tightId = cms.InputTag("tightJetIdAK8"),
        tightIdLepVeto = cms.InputTag("tightJetIdLepVetoAK8"),
      ),
)
run2_jme_2016.toModify(updatedJetsAK8WithUserData.userInts,
    looseId = cms.InputTag("looseJetIdAK8"),
)


finalJets = cms.EDFilter("PATJetRefSelector",
    src = cms.InputTag("updatedJetsWithUserData"),
    cut = cms.string("pt > 15")
)

finalJetsAK8 = cms.EDFilter("PATJetRefSelector",
    src = cms.InputTag("updatedJetsAK8WithUserData"),
    cut = cms.string("pt > 170")
)

lepInJetVars = cms.EDProducer("LepInJetProducer",
    src = cms.InputTag("updatedJetsAK8WithUserData"),
    srcEle = cms.InputTag("finalElectrons"),
    srcMu = cms.InputTag("finalMuons")
)



##################### Tables for final output and docs ##########################



jetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("linkedObjects","jets"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("Jet"),
    doc  = cms.string("slimmedJets, i.e. ak4 PFJets CHS with JECs applied, after basic selection (" + finalJets.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    externalVariables = cms.PSet(
        bRegCorr = ExtVar(cms.InputTag("bjetNN:corr"),float, doc="pt correction for b-jet energy regression",precision=10),
        bRegRes = ExtVar(cms.InputTag("bjetNN:res"),float, doc="res on pt corrected with b-jet regression",precision=6),
        cRegCorr = ExtVar(cms.InputTag("cjetNN:corr"),float, doc="pt correction for c-jet energy regression",precision=10),
        cRegRes = ExtVar(cms.InputTag("cjetNN:res"),float, doc="res on pt corrected with c-jet regression",precision=6),
    ),
    variables = cms.PSet(P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        nMuons = Var("?hasOverlaps('muons')?overlaps('muons').size():0", int, doc="number of muons in the jet"),
        muonIdx1 = Var("?overlaps('muons').size()>0?overlaps('muons')[0].key():-1", int, doc="index of first matching muon"),
        muonIdx2 = Var("?overlaps('muons').size()>1?overlaps('muons')[1].key():-1", int, doc="index of second matching muon"),
        electronIdx1 = Var("?overlaps('electrons').size()>0?overlaps('electrons')[0].key():-1", int, doc="index of first matching electron"),
        electronIdx2 = Var("?overlaps('electrons').size()>1?overlaps('electrons')[1].key():-1", int, doc="index of second matching electron"),
        nElectrons = Var("?hasOverlaps('electrons')?overlaps('electrons').size():0", int, doc="number of electrons in the jet"),
        btagCMVA = Var("bDiscriminator('pfCombinedMVAV2BJetTags')",float,doc="CMVA V2 btag discriminator",precision=10),
        btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="DeepCSV b+bb tag discriminator",precision=10),
        btagDeepFlavB = Var("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')",float,doc="DeepFlavour b+bb+lepb tag discriminator",precision=10),
        btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
        btagDeepC = Var("bDiscriminator('pfDeepCSVJetTags:probc')",float,doc="DeepCSV charm btag discriminator",precision=10),
        btagDeepFlavC = Var("bDiscriminator('pfDeepFlavourJetTags:probc')",float,doc="DeepFlavour charm tag discriminator",precision=10),
        puIdDisc = Var("userFloat('puId102XDisc')",float,doc="Pilup ID discriminant with 102X (2018) training",precision=10),
        puId = Var("userInt('pileupJetId:fullId')",int,doc="Pilup ID flags with 80X (2016) training"),
        jetId = Var("userInt('tightId')*2+4*userInt('tightIdLepVeto')",int,doc="Jet ID flags bit1 is loose (always false in 2017 since it does not exist), bit2 is tight, bit3 is tightLepVeto"),
        qgl = Var("userFloat('qgl')",float,doc="Quark vs Gluon likelihood discriminator",precision=10),
        nConstituents = Var("numberOfDaughters()",int,doc="Number of particles in the jet"),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        chHEF = Var("chargedHadronEnergyFraction()", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergyFraction()", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("chargedEmEnergyFraction()", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("neutralEmEnergyFraction()", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        muEF = Var("muonEnergyFraction()", float, doc="muon Energy Fraction", precision= 6),
        chFPV0EF = Var("userFloat('chFPV0EF')", float, doc="charged fromPV==0 Energy Fraction (energy excluded from CHS jets). Previously called betastar.", precision= 6),
        chFPV1EF = Var("userFloat('chFPV1EF')", float, doc="charged fromPV==1 Energy Fraction (component of the total charged Energy Fraction).", precision= 6),
        chFPV2EF = Var("userFloat('chFPV2EF')", float, doc="charged fromPV==2 Energy Fraction (component of the total charged Energy Fraction).", precision= 6),
        chFPV3EF = Var("userFloat('chFPV3EF')", float, doc="charged fromPV==3 Energy Fraction (component of the total charged Energy Fraction).", precision= 6),
    )
)

#jets are not as precise as muons
jetTable.variables.pt.precision=10

### Era dependent customization
run2_jme_2016.toModify( jetTable.variables, jetId = Var("userInt('tightIdLepVeto')*4+userInt('tightId')*2+userInt('looseId')",int,doc="Jet ID flags bit1 is loose, bit2 is tight, bit3 is tightLepVeto"))
run2_jme_2016.toModify( jetTable.variables, puIdDisc = Var("userFloat('pileupJetId:fullDiscriminant')",float,doc="Pilup ID discriminant with 80X (2016) training",precision=10))
run2_jme_2017.toModify( jetTable.variables, puIdDisc = Var("userFloat('puId94XDisc')", float,doc="Pilup ID discriminant with 94X (2017) training",precision=10))

bjetNN= cms.EDProducer("BJetEnergyRegressionMVA",
    backend = cms.string("TF"),
    src = cms.InputTag("linkedObjects","jets"),
    pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    svsrc = cms.InputTag("slimmedSecondaryVertices"),
    rhosrc = cms.InputTag("fixedGridRhoFastjetAll"),

    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/breg_training_2018.pb"),
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
    #Jet_leptonPt = cms.string("userFloat('leptonPt')"),
    Jet_leptonPtRel = cms.string("userFloat('leptonPtRelv0')"),
    Jet_leptonPtRelInv = cms.string("userFloat('leptonPtRelInvv0')*jecFactor('Uncorrected')"),
    Jet_leptonDeltaR = cms.string("userFloat('leptonDeltaR')"),
    #Jet_leptonPdgId = cms.string("userInt('leptonPdgId')"),
    Jet_neHEF = cms.string("neutralHadronEnergyFraction()"),
    Jet_neEmEF = cms.string("neutralEmEnergyFraction()"),
    Jet_chHEF = cms.string("chargedHadronEnergyFraction()"),
    Jet_chEmEF = cms.string("chargedEmEnergyFraction()"),
    isMu = cms.string("?abs(userInt('leptonPdgId'))==13?1:0"),
    isEle = cms.string("?abs(userInt('leptonPdgId'))==11?1:0"),
    isOther = cms.string("?userInt('leptonPdgId')==0?1:0"),
    ),
     inputTensorName = cms.string("ffwd_inp"),
     outputTensorName = cms.string("ffwd_out/BiasAdd"),
     outputNames = cms.vstring(["corr","res"]),
     outputFormulas = cms.vstring(["at(0)*0.27912887930870056+1.0545977354049683","0.5*(at(2)-at(1))*0.27912887930870056"]),
     nThreads = cms.uint32(1),
     singleThreadPool = cms.string("no_threads"),
)

cjetNN= cms.EDProducer("BJetEnergyRegressionMVA",
    backend = cms.string("TF"),
    src = cms.InputTag("linkedObjects","jets"),
    pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    svsrc = cms.InputTag("slimmedSecondaryVertices"),
    rhosrc = cms.InputTag("fixedGridRhoFastjetAll"),

    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/creg_training_2018.pb"),
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
    #Jet_leptonPt = cms.string("userFloat('leptonPt')"),
    Jet_leptonPtRel = cms.string("userFloat('leptonPtRelv0')"),
    Jet_leptonPtRelInv = cms.string("userFloat('leptonPtRelInvv0')*jecFactor('Uncorrected')"),
    Jet_leptonDeltaR = cms.string("userFloat('leptonDeltaR')"),
    #Jet_leptonPdgId = cms.string("userInt('leptonPdgId')"),
    Jet_neHEF = cms.string("neutralHadronEnergyFraction()"),
    Jet_neEmEF = cms.string("neutralEmEnergyFraction()"),
    Jet_chHEF = cms.string("chargedHadronEnergyFraction()"),
    Jet_chEmEF = cms.string("chargedEmEnergyFraction()"),
    isMu = cms.string("?abs(userInt('leptonPdgId'))==13?1:0"),
    isEle = cms.string("?abs(userInt('leptonPdgId'))==11?1:0"),
    isOther = cms.string("?userInt('leptonPdgId')==0?1:0"),
    ),
     inputTensorName = cms.string("ffwd_inp"),
     outputTensorName = cms.string("ffwd_out/BiasAdd"),
     outputNames = cms.vstring(["corr","res"]),
     outputFormulas = cms.vstring(["at(0)*0.24325256049633026+0.993854820728302","0.5*(at(2)-at(1))*0.24325256049633026"]),
     nThreads = cms.uint32(1),
     singleThreadPool = cms.string("no_threads"),
)


##### Soft Activity tables
saJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("softActivityJets"),
    cut = cms.string(""),
    maxLen = cms.uint32(6),
    name = cms.string("SoftActivityJet"),
    doc  = cms.string("jets clustered from charged candidates compatible with primary vertex (" + chsForSATkJets.cut.value()+")"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    variables = cms.PSet(P3Vars,
  )
)

saJetTable.variables.pt.precision=10
saJetTable.variables.eta.precision=8
saJetTable.variables.phi.precision=8

saTable = cms.EDProducer("GlobalVariablesTableProducer",
    variables = cms.PSet(
        SoftActivityJetHT = ExtVar( cms.InputTag("softActivityJets"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt, pt>1" ),
        SoftActivityJetHT10 = ExtVar( cms.InputTag("softActivityJets10"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt , pt >10"  ),
        SoftActivityJetHT5 = ExtVar( cms.InputTag("softActivityJets5"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt, pt>5"  ),
        SoftActivityJetHT2 = ExtVar( cms.InputTag("softActivityJets2"), "candidatescalarsum", doc = "scalar sum of soft activity jet pt, pt >2"  ),
        SoftActivityJetNjets10 = ExtVar( cms.InputTag("softActivityJets10"), "candidatesize", doc = "number of soft activity jet pt, pt >2"  ),
        SoftActivityJetNjets5 = ExtVar( cms.InputTag("softActivityJets5"), "candidatesize", doc = "number of soft activity jet pt, pt >5"  ),
        SoftActivityJetNjets2 = ExtVar( cms.InputTag("softActivityJets2"), "candidatesize", doc = "number of soft activity jet pt, pt >10"  ),

    )
)



## BOOSTED STUFF #################
fatJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("finalJetsAK8"),
    cut = cms.string(" pt > 170"), #probably already applied in miniaod
    name = cms.string("FatJet"),
    doc  = cms.string("slimmedJetsAK8, i.e. ak8 fat jets for boosted analysis"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    variables = cms.PSet(P4Vars,
        jetId = Var("userInt('tightId')*2+4*userInt('tightIdLepVeto')",int,doc="Jet ID flags bit1 is loose (always false in 2017 since it does not exist), bit2 is tight, bit3 is tightLepVeto"),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        tau1 = Var("userFloat('NjettinessAK8Puppi:tau1')",float, doc="Nsubjettiness (1 axis)",precision=10),
        tau2 = Var("userFloat('NjettinessAK8Puppi:tau2')",float, doc="Nsubjettiness (2 axis)",precision=10),
        tau3 = Var("userFloat('NjettinessAK8Puppi:tau3')",float, doc="Nsubjettiness (3 axis)",precision=10),
        tau4 = Var("userFloat('NjettinessAK8Puppi:tau4')",float, doc="Nsubjettiness (4 axis)",precision=10),
        n2b1 = Var("userFloat('ak8PFJetsPuppiSoftDropValueMap:nb1AK8PuppiSoftDropN2')", float, doc="N2 with beta=1", precision=10),
        n3b1 = Var("userFloat('ak8PFJetsPuppiSoftDropValueMap:nb1AK8PuppiSoftDropN3')", float, doc="N3 with beta=1", precision=10),
        msoftdrop = Var("groomedMass('SoftDropPuppi')",float, doc="Corrected soft drop mass with PUPPI",precision=10),
        btagCMVA = Var("bDiscriminator('pfCombinedMVAV2BJetTags')",float,doc="CMVA V2 btag discriminator",precision=10),
        btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="DeepCSV b+bb tag discriminator",precision=10),
        btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
        btagHbb = Var("bDiscriminator('pfBoostedDoubleSecondaryVertexAK8BJetTags')",float,doc="Higgs to BB tagger discriminator",precision=10),
        btagDDBvL_noMD = Var("bDiscriminator('pfDeepDoubleBvLJetTags:probHbb')",float,doc="DeepDoubleX discriminator (no mass-decorrelation) for H(Z)->bb vs QCD",precision=10),
        btagDDCvL_noMD = Var("bDiscriminator('pfDeepDoubleCvLJetTags:probHcc')",float,doc="DeepDoubleX discriminator (no mass-decorrelation) for H(Z)->cc vs QCD",precision=10),
        btagDDCvB_noMD = Var("bDiscriminator('pfDeepDoubleCvBJetTags:probHcc')",float,doc="DeepDoubleX discriminator (no mass-decorrelation) for H(Z)->cc vs H(Z)->bb",precision=10),
        btagDDBvL = Var("bDiscriminator('pfMassIndependentDeepDoubleBvLJetTags:probHbb')",float,doc="DeepDoubleX (mass-decorrelated) discriminator for H(Z)->bb vs QCD",precision=10),
        btagDDCvL = Var("bDiscriminator('pfMassIndependentDeepDoubleCvLJetTags:probHcc')",float,doc="DeepDoubleX (mass-decorrelated) discriminator for H(Z)->cc vs QCD",precision=10),
        btagDDCvB = Var("bDiscriminator('pfMassIndependentDeepDoubleCvBJetTags:probHcc')",float,doc="DeepDoubleX (mass-decorrelated) discriminator for H(Z)->cc vs H(Z)->bb",precision=10),
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
        subJetIdx1 = Var("?nSubjetCollections()>0 && subjets('SoftDropPuppi').size()>0?subjets('SoftDropPuppi')[0].key():-1", int,
		     doc="index of first subjet"),
        subJetIdx2 = Var("?nSubjetCollections()>0 && subjets('SoftDropPuppi').size()>1?subjets('SoftDropPuppi')[1].key():-1", int,
		     doc="index of second subjet"),

#        btagDeepC = Var("bDiscriminator('pfDeepCSVJetTags:probc')",float,doc="CMVA V2 btag discriminator",precision=10),
#puIdDisc = Var("userFloat('pileupJetId:fullDiscriminant')",float,doc="Pilup ID discriminant",precision=10),
#        nConstituents = Var("numberOfDaughters()",int,doc="Number of particles in the jet"),
#        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
    ),
    externalVariables = cms.PSet(
        lsf3 = ExtVar(cms.InputTag("lepInJetVars:lsf3"),float, doc="Lepton Subjet Fraction (3 subjets)",precision=10),
        muonIdx3SJ = ExtVar(cms.InputTag("lepInJetVars:muIdx3SJ"),int, doc="index of muon matched to jet"),
        electronIdx3SJ = ExtVar(cms.InputTag("lepInJetVars:eleIdx3SJ"),int,doc="index of electron matched to jet"),
    )
)
### Era dependent customization
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables, msoftdrop_chs = Var("userFloat('ak8PFJetsCHSSoftDropMass')",float, doc="Legacy uncorrected soft drop mass with CHS",precision=10))
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.tau1, expr = cms.string("userFloat(\'ak8PFJetsPuppiValueMap:NjettinessAK8PuppiTau1\')"),)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.tau2, expr = cms.string("userFloat(\'ak8PFJetsPuppiValueMap:NjettinessAK8PuppiTau2\')"),)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.tau3, expr = cms.string("userFloat(\'ak8PFJetsPuppiValueMap:NjettinessAK8PuppiTau3\')"),)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables, tau4 = None)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables, n2b1 = None)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables, n3b1 = None)
run2_jme_2016.toModify( fatJetTable.variables, jetId = Var("userInt('tightId')*2+userInt('looseId')",int,doc="Jet ID flags bit1 is loose, bit2 is tight"))

run2_jme_2016.toModify( bjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/breg_training_2016.pb") )
run2_jme_2016.toModify( bjetNN,outputFormulas = cms.vstring(["at(0)*0.31976690888404846+1.047176718711853","0.5*(at(2)-at(1))*0.31976690888404846"]))

run2_jme_2017.toModify( bjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/breg_training_2017.pb") )
run2_jme_2017.toModify( bjetNN,outputFormulas = cms.vstring(["at(0)*0.28225210309028625+1.055067777633667","0.5*(at(2)-at(1))*0.28225210309028625"]))

run2_jme_2016.toModify( cjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/creg_training_2016.pb") )
run2_jme_2016.toModify( cjetNN,outputFormulas = cms.vstring(["at(0)*0.28862622380256653+0.9908722639083862","0.5*(at(2)-at(1))*0.28862622380256653"]))

run2_jme_2017.toModify( cjetNN, weightFile = cms.FileInPath("PhysicsTools/NanoAOD/data/creg_training_2017.pb") )
run2_jme_2017.toModify( cjetNN,outputFormulas = cms.vstring(["at(0)*0.24718524515628815+0.9927206635475159","0.5*(at(2)-at(1))*0.24718524515628815"]))



subJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked","SubJets"),
    cut = cms.string(""), #probably already applied in miniaod
    name = cms.string("SubJet"),
    doc  = cms.string("slimmedJetsAK8, i.e. ak8 fat jets for boosted analysis"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    variables = cms.PSet(P4Vars,
        btagCMVA = Var("bDiscriminator('pfCombinedMVAV2BJetTags')",float,doc="CMVA V2 btag discriminator",precision=10),
        btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="DeepCSV b+bb tag discriminator",precision=10),
        btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),                 
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

run2_miniAOD_80XLegacy.toModify( subJetTable.variables, tau1 = None)
run2_miniAOD_80XLegacy.toModify( subJetTable.variables, tau2 = None)
run2_miniAOD_80XLegacy.toModify( subJetTable.variables, tau3 = None)
run2_miniAOD_80XLegacy.toModify( subJetTable.variables, tau4 = None)
run2_miniAOD_80XLegacy.toModify( subJetTable.variables, n2b1 = None)
run2_miniAOD_80XLegacy.toModify( subJetTable.variables, n3b1 = None)
run2_miniAOD_80XLegacy.toModify( subJetTable.variables, btagCMVA = None, btagDeepB = None)


corrT1METJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("corrT1METJets"),
    cut = cms.string(""),
    name = cms.string("CorrT1METJet"),
    doc  = cms.string("Additional low-pt jets for Type-1 MET re-correction"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    variables = cms.PSet(
        rawPt = Var("pt()*jecFactor('Uncorrected')",float,precision=10),
        eta  = Var("eta",  float,precision=12),
        phi = Var("phi", float, precision=12),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
    )
)



## MC STUFF ######################
jetMCTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("linkedObjects","jets"),
    cut = cms.string(""), #we should not filter on cross linked collections
    name = cms.string("Jet"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(True), # this is an extension  table for the jets
    variables = cms.PSet(
        partonFlavour = Var("partonFlavour()", int, doc="flavour from parton matching"),
        hadronFlavour = Var("hadronFlavour()", int, doc="flavour from hadron ghost clustering"),
        genJetIdx = Var("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().key():-1", int, doc="index of matched gen jet"),
    )
)
genJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedGenJets"),
    cut = cms.string("pt > 10"),
    name = cms.string("GenJet"),
    doc  = cms.string("slimmedGenJets, i.e. ak4 Jets made with visible genparticles"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the genjets
    variables = cms.PSet(P4Vars,
	#anything else?
    )
)
patJetPartons = cms.EDProducer('HadronAndPartonSelector',
    src = cms.InputTag("generator"),
    particles = cms.InputTag("prunedGenParticles"),
    partonMode = cms.string("Auto"),
    fullChainPhysPartons = cms.bool(True)
)
genJetFlavourAssociation = cms.EDProducer("JetFlavourClustering",
    jets = genJetTable.src,
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    partons = cms.InputTag("patJetPartons","physicsPartons"),
    leptons = cms.InputTag("patJetPartons","leptons"),
    jetAlgorithm = cms.string("AntiKt"),
    rParam = cms.double(0.4),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)
genJetFlavourTable = cms.EDProducer("GenJetFlavourTableProducer",
    name = genJetTable.name,
    src = genJetTable.src,
    cut = genJetTable.cut,
    deltaR = cms.double(0.1),
    jetFlavourInfos = cms.InputTag("slimmedGenJetsFlavourInfos"),
)

genJetAK8Table = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedGenJetsAK8"),
    cut = cms.string("pt > 100."),
    name = cms.string("GenJetAK8"),
    doc  = cms.string("slimmedGenJetsAK8, i.e. ak8 Jets made with visible genparticles"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the genjets
    variables = cms.PSet(P4Vars,
	#anything else?
    )
)
genJetAK8FlavourAssociation = cms.EDProducer("JetFlavourClustering",
    jets = genJetAK8Table.src,
    bHadrons = cms.InputTag("patJetPartons","bHadrons"),
    cHadrons = cms.InputTag("patJetPartons","cHadrons"),
    partons = cms.InputTag("patJetPartons","physicsPartons"),
    leptons = cms.InputTag("patJetPartons","leptons"),
    jetAlgorithm = cms.string("AntiKt"),
    rParam = cms.double(0.8),
    ghostRescaling = cms.double(1e-18),
    hadronFlavourHasPriority = cms.bool(False)
)
genJetAK8FlavourTable = cms.EDProducer("GenJetFlavourTableProducer",
    name = genJetAK8Table.name,
    src = genJetAK8Table.src,
    cut = genJetAK8Table.cut,
    deltaR = cms.double(0.1),
    jetFlavourInfos = cms.InputTag("genJetAK8FlavourAssociation"),
)
fatJetMCTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = fatJetTable.src,
    cut = fatJetTable.cut,
    name = fatJetTable.name,
    singleton = cms.bool(False),
    extension = cms.bool(True),
    variables = cms.PSet(
        nBHadrons = Var("jetFlavourInfo().getbHadrons().size()", "uint8", doc="number of b-hadrons"),
        nCHadrons = Var("jetFlavourInfo().getcHadrons().size()", "uint8", doc="number of c-hadrons"),
        hadronFlavour = Var("hadronFlavour()", int, doc="flavour from hadron ghost clustering"),
        genJetAK8Idx = Var("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().key():-1", int, doc="index of matched gen AK8 jet"),
    )
)

genSubJetAK8Table = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedGenJetsAK8SoftDropSubJets"),
    cut = cms.string(""),  ## These don't get a pt cut, but in miniAOD only subjets from fat jets with pt > 100 are kept
    name = cms.string("SubGenJetAK8"),
    doc  = cms.string("slimmedGenJetsAK8SoftDropSubJets, i.e. subjets of ak8 Jets made with visible genparticles"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the genjets
    variables = cms.PSet(P4Vars,
	#anything else?
    )
)
subjetMCTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = subJetTable.src,
    cut = subJetTable.cut,
    name = subJetTable.name,
    singleton = cms.bool(False),
    extension = cms.bool(True),
    variables = cms.PSet(
        nBHadrons = Var("jetFlavourInfo().getbHadrons().size()", "uint8", doc="number of b-hadrons"),
        nCHadrons = Var("jetFlavourInfo().getcHadrons().size()", "uint8", doc="number of c-hadrons"),
    )
)

### Era dependent customization
run2_miniAOD_80XLegacy.toModify( genJetFlavourTable, jetFlavourInfos = cms.InputTag("genJetFlavourAssociation"),)

from RecoJets.JetProducers.QGTagger_cfi import  QGTagger
qgtagger=QGTagger.clone(srcJets="updatedJets",srcVertexCollection="offlineSlimmedPrimaryVertices")

from RecoJets.JetProducers.PileupJetID_cfi import pileupJetId, _chsalgos_94x, _chsalgos_102x
pileupJetId94X=pileupJetId.clone(jets="updatedJets",algos = cms.VPSet(_chsalgos_94x),inputIsCorrected=True,applyJec=False,vertexes="offlineSlimmedPrimaryVertices")
pileupJetId102X=pileupJetId.clone(jets="updatedJets",algos = cms.VPSet(_chsalgos_102x),inputIsCorrected=True,applyJec=False,vertexes="offlineSlimmedPrimaryVertices")

#before cross linking
jetSequence = cms.Sequence(jetCorrFactorsNano+updatedJets+tightJetId+tightJetIdLepVeto+bJetVars+qgtagger+jercVars+pileupJetId94X+pileupJetId102X+updatedJetsWithUserData+jetCorrFactorsAK8+updatedJetsAK8+tightJetIdAK8+tightJetIdLepVetoAK8+updatedJetsAK8WithUserData+chsForSATkJets+softActivityJets+softActivityJets2+softActivityJets5+softActivityJets10+finalJets+finalJetsAK8)


_jetSequence_2016 = jetSequence.copy()
_jetSequence_2016.insert(_jetSequence_2016.index(tightJetId), looseJetId)
_jetSequence_2016.insert(_jetSequence_2016.index(tightJetIdAK8), looseJetIdAK8)
run2_jme_2016.toReplaceWith(jetSequence, _jetSequence_2016)

#after lepton collections have been run
jetLepSequence = cms.Sequence(lepInJetVars)

#after cross linkining
jetTables = cms.Sequence(bjetNN+cjetNN+jetTable+fatJetTable+subJetTable+saJetTable+saTable)

#MC only producers and tables
jetMC = cms.Sequence(jetMCTable+genJetTable+patJetPartons+genJetFlavourTable+genJetAK8Table+genJetAK8FlavourAssociation+genJetAK8FlavourTable+fatJetMCTable+genSubJetAK8Table+subjetMCTable)
_jetMC_pre94X = jetMC.copy()
_jetMC_pre94X.insert(_jetMC_pre94X.index(genJetFlavourTable),genJetFlavourAssociation)
_jetMC_pre94X.remove(genSubJetAK8Table)
run2_miniAOD_80XLegacy.toReplaceWith(jetMC, _jetMC_pre94X)


