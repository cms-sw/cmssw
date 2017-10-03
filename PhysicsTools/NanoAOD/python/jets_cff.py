import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

from  PhysicsTools.NanoAOD.common_cff import *



##################### User floats producers, selectors ##########################
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

chsForSATkJets = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string('charge()!=0 && pvAssociationQuality()>=5 && vertexRef().key()==0'))
softActivityJets = ak4PFJets.clone(src = 'chsForSATkJets', doAreaFastjet = False, jetPtMin=1) 
softActivityJets10 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("chsForSATkJets"), cut = cms.string('pt>10'))
softActivityJets5 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("chsForSATkJets"), cut = cms.string('pt>5'))
softActivityJets2 = cms.EDFilter("CandPtrSelector", src = cms.InputTag("chsForSATkJets"), cut = cms.string('pt>2'))

looseJetId = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('WINTER16'),
			    quality = cms.string('LOOSE'),
			  ),
                          src = cms.InputTag("slimmedJets")
)
tightJetId = cms.EDProducer("PatJetIDValueMapProducer",
			  filterParams=cms.PSet(
			    version = cms.string('WINTER16'),
			    quality = cms.string('TIGHT'),
			  ),
                          src = cms.InputTag("slimmedJets")
)


slimmedJetsWithUserData = cms.EDProducer("PATJetUserDataEmbedder",
     src = cms.InputTag("slimmedJets"),
     userInts = cms.PSet(
        tightId = cms.InputTag("tightJetId"),
        looseId = cms.InputTag("looseJetId"),
     ),
)


from  PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
jetCorrFactors = patJetCorrFactors.clone(src='slimmedJetsWithUserData',
    levels = cms.vstring('L1FastJet',
        'L2Relative',
        'L3Absolute',
	'L2L3Residual'),
    primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
)
from  PhysicsTools.PatAlgos.producersLayer1.jetUpdater_cfi import *
updatedJets = updatedPatJets.clone(
	addBTagInfo=False,
	jetSource='slimmedJetsWithUserData',
	jetCorrFactorsSource=cms.VInputTag(cms.InputTag("jetCorrFactors") ),
)

finalJets = cms.EDFilter("PATJetRefSelector",
    src = cms.InputTag("updatedJets"),
    cut = cms.string("pt > 15")
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
        bReg = ExtVar(cms.InputTag("bjetMVA"),float, doc="pt corrected with b-jet regression",precision=14),
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
	btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')",float,doc="DeepCSV b tag discriminator",precision=10),
	btagDeepBB = Var("bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="DeepCSV bb tag discriminator",precision=10),
	btagDeepC = Var("bDiscriminator('pfDeepCSVJetTags:probc')",float,doc="DeepCSV charm btag discriminator",precision=10),
#puIdDisc = Var("userFloat('pileupJetId:fullDiscriminant')",float,doc="Pilup ID discriminant",precision=10),
	puId = Var("userInt('pileupJetId:fullId')",int,doc="Pilup ID flags"),
	jetId = Var("userInt('tightId')*2+userInt('looseId')",int,doc="Jet ID flags bit1 is loose, bit2 is tight"),
	qgl = Var("userFloat('QGTagger:qgLikelihood')",float,doc="Quark vs Gluon likelihood discriminator",precision=10),
	nConstituents = Var("numberOfDaughters()",int,doc="Number of particles in the jet"),
	rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        chHEF = Var("chargedHadronEnergy()/energy()", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergy()/energy()", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("chargedEmEnergy()/energy()", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("neutralEmEnergy()/energy()", float, doc="charged Electromagnetic EnergyFraction", precision= 6),

    )
)
#jets are not as precise as muons
jetTable.variables.pt.precision=10

### Era dependent customization
run2_miniAOD_80XLegacy.toModify( jetTable.variables.qgl, expr="-1" )

bjetMVA= cms.EDProducer("BJetEnergyRegressionMVA",
    src = cms.InputTag("linkedObjects","jets"),
    pvsrc = cms.InputTag("offlineSlimmedPrimaryVertices"),
    svsrc = cms.InputTag("slimmedSecondaryVertices"),
    weightFile =  cms.FileInPath("PhysicsTools/NanoAOD/data/bjet-regression.xml"),
    name = cms.string("JetReg"),
    isClassifier = cms.bool(False),
    variablesOrder = cms.vstring(["Jet_pt","nPVs","Jet_eta","Jet_mt","Jet_leadTrackPt","Jet_leptonPtRel","Jet_leptonPt","Jet_leptonDeltaR","Jet_neHEF","Jet_neEmEF","Jet_vtxPt","Jet_vtxMass","Jet_vtx3dL","Jet_vtxNtrk","Jet_vtx3deL"]),
    variables = cms.PSet(
	Jet_pt = cms.string("pt"),
	Jet_eta = cms.string("eta"),
	Jet_mt = cms.string("mt"),
	Jet_leptonPt = cms.string("?overlaps('muons').size()>0?overlaps('muons')[0].pt():(?overlaps('electrons').size()>0?overlaps('electrons')[0].pt():0)"),
	Jet_neHEF = cms.string("neutralHadronEnergy()/energy()"),
	Jet_neEmEF = cms.string("neutralEmEnergy()/energy()"),
	Jet_leptonDeltaR = cms.string('''?overlaps('muons').size()>0?deltaR(eta,phi,overlaps('muons')[0].eta,overlaps('muons')[0].phi):
				(?overlaps('electrons').size()>0?deltaR(eta,phi,overlaps('electrons')[0].eta,overlaps('electrons')[0].phi):
				0)'''),
    )

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
    src = cms.InputTag("slimmedJetsAK8"),
    cut = cms.string(" pt > 170"), #probably already applied in miniaod
    name = cms.string("FatJet"),
    doc  = cms.string("slimmedJetsAK8, i.e. ak8 fat jets for boosted analysis"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    variables = cms.PSet(P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        tau1 = Var("userFloat('ak8PFJetsCHSValueMap:NjettinessAK8CHSTau1')",float, doc="Nsubjettiness (1 axis)",precision=10),
        tau2 = Var("userFloat('ak8PFJetsCHSValueMap:NjettinessAK8CHSTau2')",float, doc="Nsubjettiness (2 axis)",precision=10),
        tau3 = Var("userFloat('ak8PFJetsCHSValueMap:NjettinessAK8CHSTau3')",float, doc="Nsubjettiness (3 axis)",precision=10),
        msoftdrop = Var("userFloat('ak8PFJetsCHSValueMap:ak8PFJetsCHSSoftDropMass')",float, doc="Soft drop mass",precision=10),
        mpruned = Var("userFloat('ak8PFJetsCHSValueMap:ak8PFJetsCHSPrunedMass')", float, doc="Pruned mass",precision=10),

        btagCMVA = Var("bDiscriminator('pfCombinedMVAV2BJetTags')",float,doc="CMVA V2 btag discriminator",precision=10),
        btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')",float,doc="DeepCSV B btag discriminator",precision=10),
        btagDeepBB = Var("bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="DeepCSV BB btag discriminator",precision=10),
        btagHbb = Var("bDiscriminator('pfBoostedDoubleSecondaryVertexAK8BJetTags')",float,doc="Higgs to BB tagger discriminator",precision=10),

        subJetIdx1 = Var("?numberOfSourceCandidatePtrs()>0 && sourceCandidatePtr(0).numberOfSourceCandidatePtrs()>0?sourceCandidatePtr(0).key():-1", int,
		     doc="index of first subjet"),
        subJetIdx2 = Var("?numberOfSourceCandidatePtrs()>1 && sourceCandidatePtr(1).numberOfSourceCandidatePtrs()>0?sourceCandidatePtr(1).key():-1", int,
		     doc="index of second subjet"),
        subJetIdx3 = Var("?numberOfSourceCandidatePtrs()>2 && sourceCandidatePtr(2).numberOfSourceCandidatePtrs()>0?sourceCandidatePtr(2).key():-1", int,
		     doc="index of third subjet"),
	
#        btagDeepC = Var("bDiscriminator('pfDeepCSVJetTags:probc')",float,doc="CMVA V2 btag discriminator",precision=10),
#puIdDisc = Var("userFloat('pileupJetId:fullDiscriminant')",float,doc="Pilup ID discriminant",precision=10),
#        nConstituents = Var("numberOfDaughters()",int,doc="Number of particles in the jet"),
#        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
    )
)
### Era dependent customization
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.mpruned, expr = cms.string("userFloat(\'ak8PFJetsCHSPrunedMass\')"),)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.msoftdrop, expr = cms.string("userFloat(\'ak8PFJetsCHSSoftDropMass\')"),)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.tau1, expr = cms.string("userFloat(\'NjettinessAK8:tau1\')"),)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.tau2, expr = cms.string("userFloat(\'NjettinessAK8:tau2\')"),)
run2_miniAOD_80XLegacy.toModify( fatJetTable.variables.tau3, expr = cms.string("userFloat(\'NjettinessAK8:tau3\')"),)

subJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked","SubJets"),
    cut = cms.string(""), #probably already applied in miniaod
    name = cms.string("SubJet"),
    doc  = cms.string("slimmedJetsAK8, i.e. ak8 fat jets for boosted analysis"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the jets
    variables = cms.PSet(P4Vars,
        btagCMVA = Var("bDiscriminator('pfCombinedMVAV2BJetTags')",float,doc="CMVA V2 btag discriminator",precision=10),
        btagDeepB = Var("bDiscriminator('pfDeepCSVJetTags:probb')",float,doc="CMVA V2 btag discriminator",precision=10),
        btagDeepBB = Var("bDiscriminator('pfDeepCSVJetTags:probbb')",float,doc="CMVA V2 btag discriminator",precision=10),
    )
)

#jets are not as precise as muons
fatJetTable.variables.pt.precision=10







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



#before cross linking
jetSequence = cms.Sequence(looseJetId+tightJetId+slimmedJetsWithUserData+jetCorrFactors+updatedJets+chsForSATkJets+softActivityJets+softActivityJets2+softActivityJets5+softActivityJets10+finalJets)
#after cross linkining
jetTables = cms.Sequence(bjetMVA+ jetTable+fatJetTable+subJetTable+saJetTable+saTable)

#MC only producers and tables
jetMC = cms.Sequence(jetMCTable+genJetTable)

