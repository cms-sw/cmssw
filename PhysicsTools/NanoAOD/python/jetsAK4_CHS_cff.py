import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simplePATJetFlatTableProducer_cfi import simplePATJetFlatTableProducer

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


jetTable = simplePATJetFlatTableProducer.clone(
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
        btagDeepFlavB = Var("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')",float,doc="DeepJet b+bb+lepb tag discriminator",precision=10),
        btagDeepFlavCvL = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1",float,doc="DeepJet c vs uds+g discriminator",precision=10),
        btagDeepFlavCvB = Var("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1",float,doc="DeepJet c vs b+bb+lepb discriminator",precision=10),
        btagDeepFlavQG = Var("?(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1",float,doc="DeepJet g vs uds discriminator",precision=10),
        btagPNetB = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:BvsAll')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:BvsAll'):-1",float,precision=10,doc="ParticleNet b vs. udscg"),
        btagPNetCvL = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsL')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsL'):-1",float,precision=10,doc="ParticleNet c vs. udsg"),
        btagPNetCvB = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsB')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsB'):-1",float,precision=10,doc="ParticleNet c vs. b"),
        btagPNetCvNotB = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:probb')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:probc')/(1.-bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:probb')):-1",float,precision=10,doc="ParticleNet C vs notB"),
        btagPNetQvG = Var("?abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:QvsG'):bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags:QvsG')",float,precision=10,doc="ParticleNet q (udsbc) vs. g"),
        btagPNetTauVJet = Var("?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:TauVsJet')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:TauVsJet'):-1",float,precision=10,doc="ParticleNet tau vs. jet"),
        PNetRegPtRawCorr = Var("?abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptcorr'):bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptcorr')",float,precision=10,doc="ParticleNet universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT"),
        PNetRegPtRawCorrNeutrino = Var("?abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptnu'):bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptnu')",float,precision=10,doc="ParticleNet universal flavor-aware pT regression neutrino correction, relative to visible. To apply full regression, multiply raw jet pT by both PNetRegPtRawCorr and PNetRegPtRawCorrNeutrino."),
        PNetRegPtRawRes = Var("?abs(eta())<2.5?0.5*(bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptreshigh')-bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptreslow')):0.5*(bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptreshigh')-bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptreslow'))",float,precision=10,doc="ParticleNet universal flavor-aware jet pT resolution estimator, (q84 - q16)/2"),
        btagUParTAK4B = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll'):-1",float,precision=10,doc="UnifiedParT b vs. udscg"),
        btagUParTAK4CvL = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL'):-1",float,precision=10,doc="UnifiedParT c vs. udsg"),
        btagUParTAK4CvB = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB'):-1",float,precision=10,doc="UnifiedParT c vs. b"),
        btagUParTAK4CvNotB = Var("?((bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probb')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:problepb')))>0?((bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probc'))/(1.-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probb')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probbb')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:problepb'))):-1",float,precision=10,doc="UnifiedParT c vs. not b"),
        btagUParTAK4SvCB  = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsBC')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsBC'):-1",float,precision=10,doc="UnifiedParT s vs. bc"),
        btagUParTAK4SvUDG = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsUDG')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsUDG'):-1",float,precision=10,doc="UnifiedParT s vs. udg"),
        btagUParTAK4UDG = Var("bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probu')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probd')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probg')",float,precision=10,doc="UnifiedParT u+d+g raw score"),
        btagUParTAK4QvG = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG'):-1",float,precision=10,doc="UnifiedParT q (uds) vs. g"),
        btagUParTAK4TauVJet = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:TauVsJet')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:TauVsJet'):-1",float,precision=10,doc="UnifiedParT tau vs. jet"),
        btagUParTAK4Ele = Var("bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probele')",float,precision=10,doc="UnifiedParT electron raw score"),
        btagUParTAK4Mu = Var("bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probmu')",float,precision=10,doc="UnifiedParT muon raw score"),
        UParTAK4RegPtRawCorr = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr'):-1",float,precision=10,doc="UnifiedParT universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT"),
        UParTAK4RegPtRawCorrNeutrino = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu'):-1",float,precision=10,doc="UnifiedParT universal flavor-aware pT regression neutrino correction, relative to visible. To apply full regression, multiply raw jet pT by both UParTAK4RegPtRawCorr and UParTAK4RegPtRawCorrNeutrino."),
        UParTAK4RegPtRawRes = Var("?(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow'))>0?0.5*(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow')):-1",float,precision=10,doc="UnifiedParT universal flavor-aware jet pT resolution estimator, (q84 - q16)/2"),
        puIdDisc = Var("userFloat('puIdNanoDisc')", float,doc="Pileup ID discriminant with 106X (2018) training",precision=10),
        puId = Var("userInt('puIdNanoId')", "uint8", doc="Pileup ID flags with 106X (2018) training"),
        qgl = Var("?userFloat('qgl')>0?userFloat('qgl'):-1",float,doc="Quark vs Gluon likelihood discriminator",precision=10),
        hfsigmaEtaEta = Var("userFloat('hfJetShowerShape:sigmaEtaEta')",float,doc="sigmaEtaEta for HF jets (noise discriminating variable)",precision=10),
        hfsigmaPhiPhi = Var("userFloat('hfJetShowerShape:sigmaPhiPhi')",float,doc="sigmaPhiPhi for HF jets (noise discriminating variable)",precision=10),
        hfcentralEtaStripSize = Var("userInt('hfJetShowerShape:centralEtaStripSize')", int, doc="eta size of the central tower strip in HF (noise discriminating variable) "),
        hfadjacentEtaStripsSize = Var("userInt('hfJetShowerShape:adjacentEtaStripsSize')", int, doc="eta size of the strips next to the central tower strip in HF (noise discriminating variable) "),
        nConstituents = Var("numberOfDaughters()","uint8",doc="Number of particles in the jet"),
        chMultiplicity = Var("chargedMultiplicity()","uint8",doc="Number of charged particles in the jet"),
        neMultiplicity = Var("neutralMultiplicity()","uint8",doc="Number of neutral particles in the jet"),
        rawFactor = Var("1.-jecFactor('Uncorrected')",float,doc="1 - Factor to get back to raw pT",precision=6),
        chHEF = Var("chargedHadronEnergyFraction()", float, doc="charged Hadron Energy Fraction", precision=10),
        neHEF = Var("neutralHadronEnergyFraction()", float, doc="neutral Hadron Energy Fraction", precision=10),
        chEmEF = Var("chargedEmEnergyFraction()", float, doc="charged Electromagnetic Energy Fraction", precision=10),
        neEmEF = Var("neutralEmEnergyFraction()", float, doc="neutral Electromagnetic Energy Fraction", precision=10),
        hfHEF = Var("HFHadronEnergyFraction()",float,doc="hadronic Energy Fraction in HF",precision=10),
        hfEmEF = Var("HFEMEnergyFraction()",float,doc="electromagnetic Energy Fraction in HF",precision=10),
        muEF = Var("muonEnergyFraction()", float, doc="muon Energy Fraction", precision=10),
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

run2_nanoAOD_ANY.toModify(
    jetTable.variables,
    btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
    btagDeepB = Var("?(bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'))>=0?bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'):-1",float,doc="DeepCSV b+bb tag discriminator",precision=10),
    btagDeepCvL = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probudsg')):-1", float,doc="DeepCSV c vs udsg discriminator",precision=10),
    btagDeepCvB = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')):-1",float,doc="DeepCSV c vs b+bb discriminator",precision=10),
    # Remove for V9
    chMultiplicity = None,
    neMultiplicity = None,
    hfHEF = None,
    hfEmEF = None
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
    variables = cms.VPSet(
        cms.PSet( name = cms.string("Jet_pt"), expr = cms.string("pt*jecFactor('Uncorrected')")),
        cms.PSet( name = cms.string("Jet_eta"), expr = cms.string("eta")),
        cms.PSet( name = cms.string("rho")),
        cms.PSet( name = cms.string("Jet_mt"), expr = cms.string("mt*jecFactor('Uncorrected')")),
        cms.PSet( name = cms.string("Jet_leadTrackPt"), expr = cms.string("userFloat('leadTrackPt')")),
        cms.PSet( name = cms.string("Jet_leptonPtRel"), expr = cms.string("userFloat('leptonPtRelv0')")),
        cms.PSet( name = cms.string("Jet_leptonDeltaR"), expr = cms.string("userFloat('leptonDeltaR')")),
        cms.PSet( name = cms.string("Jet_neHEF"), expr = cms.string("neutralHadronEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_neEmEF"), expr = cms.string("neutralEmEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_vtxPt"), expr = cms.string("userFloat('vtxPt')")),
        cms.PSet( name = cms.string("Jet_vtxMass"), expr = cms.string("userFloat('vtxMass')")),
        cms.PSet( name = cms.string("Jet_vtx3dL"), expr = cms.string("userFloat('vtx3dL')")),
        cms.PSet( name = cms.string("Jet_vtxNtrk"), expr = cms.string("userInt('vtxNtrk')")),
        cms.PSet( name = cms.string("Jet_vtx3deL"), expr = cms.string("userFloat('vtx3deL')")),
        cms.PSet( name = cms.string("Jet_numDaughters_pt03")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_chHEF"), expr = cms.string("chargedHadronEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_chEmEF"), expr = cms.string("chargedEmEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_leptonPtRelInv"), expr = cms.string("userFloat('leptonPtRelInvv0')*jecFactor('Uncorrected')")),
        cms.PSet( name = cms.string("isEle"), expr = cms.string("?abs(userInt('leptonPdgId'))==11?1:0")),
        cms.PSet( name = cms.string("isMu"), expr = cms.string("?abs(userInt('leptonPdgId'))==13?1:0")),
        cms.PSet( name = cms.string("isOther"), expr = cms.string("?userInt('leptonPdgId')==0?1:0")),
        cms.PSet( name = cms.string("Jet_mass"), expr = cms.string("mass*jecFactor('Uncorrected')")),
        cms.PSet( name = cms.string("Jet_ptd"), expr = cms.string("userFloat('ptD')"))
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
    variables = cms.VPSet(
        cms.PSet( name = cms.string("Jet_pt"), expr = cms.string("pt*jecFactor('Uncorrected')")),
	cms.PSet( name = cms.string("Jet_eta"), expr = cms.string("eta")),
        cms.PSet( name = cms.string("rho")),
        cms.PSet( name = cms.string("Jet_mt"), expr = cms.string("mt*jecFactor('Uncorrected')")),
        cms.PSet( name = cms.string("Jet_leadTrackPt"), expr = cms.string("userFloat('leadTrackPt')")),
        cms.PSet( name = cms.string("Jet_leptonPtRel"), expr = cms.string("userFloat('leptonPtRelv0')")),
        cms.PSet( name = cms.string("Jet_leptonDeltaR"), expr = cms.string("userFloat('leptonDeltaR')")),
        cms.PSet( name = cms.string("Jet_neHEF"), expr = cms.string("neutralHadronEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_neEmEF"), expr = cms.string("neutralEmEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_vtxPt"), expr = cms.string("userFloat('vtxPt')")),
        cms.PSet( name = cms.string("Jet_vtxMass"), expr = cms.string("userFloat('vtxMass')")),
        cms.PSet( name = cms.string("Jet_vtx3dL"), expr = cms.string("userFloat('vtx3dL')")),
        cms.PSet( name = cms.string("Jet_vtxNtrk"), expr = cms.string("userInt('vtxNtrk')")),
        cms.PSet( name = cms.string("Jet_vtx3deL"), expr = cms.string("userFloat('vtx3deL')")),
        cms.PSet( name = cms.string("Jet_numDaughters_pt03")),
        cms.PSet( name = cms.string("Jet_chEmEF"), expr = cms.string("chargedEmEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_chHEF"), expr = cms.string("chargedHadronEnergyFraction()")),
        cms.PSet( name = cms.string("Jet_ptd"), expr = cms.string("userFloat('ptD')")),
        cms.PSet( name = cms.string("Jet_mass"), expr = cms.string("mass*jecFactor('Uncorrected')")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_em_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_neut_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_ch_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR0_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR1_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR2_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR3_mu_Jet_rawEnergy")),
        cms.PSet( name = cms.string("Jet_energyRing_dR4_mu_Jet_rawEnergy")),
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
from RecoJets.JetProducers.PileupJetID_cfi import pileupJetId, _chsalgos_106X_UL16, _chsalgos_106X_UL16APV, _chsalgos_106X_UL17, _chsalgos_106X_UL18
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
def nanoAOD_addDeepInfoAK4CHS(process,addDeepBTag,addDeepFlavour,addParticleNet,addRobustParTAK4=False,addUnifiedParTAK4=False):
    _btagDiscriminators=[]
    if addDeepBTag:
        print("Updating process to run DeepCSV btag")
        _btagDiscriminators += ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb','pfDeepCSVJetTags:probc']
    if addDeepFlavour:
        print("Updating process to run DeepFlavour btag")
        _btagDiscriminators += ['pfDeepFlavourJetTags:probb','pfDeepFlavourJetTags:probbb','pfDeepFlavourJetTags:problepb','pfDeepFlavourJetTags:probc']
    if addParticleNet:
        print("Updating process to run ParticleNetAK4")
        from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll as pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll
        from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll as pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll
        _btagDiscriminators += pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll
        _btagDiscriminators += pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll
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
    nanoAOD_addParticleNet_switch = cms.untracked.bool(False),
    nanoAOD_addRobustParTAK4Tag_switch = cms.untracked.bool(False),
    nanoAOD_addUnifiedParTAK4Tag_switch = cms.untracked.bool(False)
)

################################################
## DeepInfoAK4CHS:End
#################################################

#
# ML-based FastSim refinement
#
from Configuration.Eras.Modifier_fastSim_cff import fastSim
def nanoAOD_refineFastSim_bTagDeepFlav(process):

    fastSim.toModify( process.jetTable.variables,
      btagDeepFlavBunrefined = process.jetTable.variables.btagDeepFlavB.clone(),
      btagDeepFlavCvBunrefined = process.jetTable.variables.btagDeepFlavCvB.clone(),
      btagDeepFlavCvLunrefined = process.jetTable.variables.btagDeepFlavCvL.clone(),
      btagDeepFlavQGunrefined = process.jetTable.variables.btagDeepFlavQG.clone(),
    )

    fastSim.toModify( process.jetTable.variables,
      btagDeepFlavB = None,
      btagDeepFlavCvB = None,
      btagDeepFlavCvL = None,
      btagDeepFlavQG = None,
    )

    fastSim.toModify( process.jetTable.externalVariables,
      btagDeepFlavB = ExtVar(cms.InputTag("btagDeepFlavRefineNN:btagDeepFlavBrefined"), float, doc="DeepJet b+bb+lepb tag discriminator", precision=10),
      btagDeepFlavCvB = ExtVar(cms.InputTag("btagDeepFlavRefineNN:btagDeepFlavCvBrefined"), float, doc="DeepJet c vs b+bb+lepb discriminator", precision=10),
      btagDeepFlavCvL = ExtVar(cms.InputTag("btagDeepFlavRefineNN:btagDeepFlavCvLrefined"), float, doc="DeepJet c vs uds+g discriminator", precision=10),
      btagDeepFlavQG = ExtVar(cms.InputTag("btagDeepFlavRefineNN:btagDeepFlavQGrefined"), float, doc="DeepJet g vs uds discriminator", precision=10),
    )

    process.btagDeepFlavRefineNN= cms.EDProducer("JetBaseMVAValueMapProducer",
        backend = cms.string("ONNX"),
        batch_eval = cms.bool(True),
        disableONNXGraphOpt = cms.bool(True),

        src = cms.InputTag("linkedObjects","jets"),

        weightFile=cms.FileInPath("PhysicsTools/NanoAOD/data/btagDeepFlavRefineNN_CHS.onnx"),
        name = cms.string("btagDeepFlavRefineNN"),

        variables = cms.VPSet(
            cms.PSet( name = cms.string("GenJet_pt"), expr = cms.string("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().pt():pt")),
            cms.PSet( name = cms.string("GenJet_eta"), expr = cms.string("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().eta():eta")),
            cms.PSet( name = cms.string("Jet_hadronFlavour"), expr = cms.string("hadronFlavour()")),
            cms.PSet( name = cms.string("Jet_btagDeepFlavB"), expr = cms.string("bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')")),
            cms.PSet( name = cms.string("Jet_btagDeepFlavCvB"), expr = cms.string("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1")),
            cms.PSet( name = cms.string("Jet_btagDeepFlavCvL"), expr = cms.string("?(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1")),
            cms.PSet( name = cms.string("Jet_btagDeepFlavQG"), expr = cms.string("?(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1")),
            ),
         inputTensorName = cms.string("input"),
         outputTensorName = cms.string("output"),
         outputNames = cms.vstring(["btagDeepFlavBrefined","btagDeepFlavCvBrefined","btagDeepFlavCvLrefined","btagDeepFlavQGrefined"]),
         outputFormulas = cms.vstring(["at(0)","at(1)","at(2)","at(3)"]),
    )

    fastSim.toModify(process.jetTablesTask, process.jetTablesTask.add(process.btagDeepFlavRefineNN))

    return process


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

corrT1METJetTable = simplePATJetFlatTableProducer.clone(
    src = finalJets.src,
    cut = cms.string("pt<15 && abs(eta)<9.9"),
    name = cms.string("CorrT1METJet"),
    doc  = cms.string("Additional low-pt ak4 CHS jets for Type-1 MET re-correction"),
    variables = cms.PSet(
        rawPt = Var("pt()*jecFactor('Uncorrected')",float,precision=10),
        rawMass = Var("mass()*jecFactor('Uncorrected')",float,precision=10),
        eta  = Var("eta",  float,precision=12),
        phi = Var("phi", float, precision=12),
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        EmEF = Var("chargedEmEnergyFraction()+neutralEmEnergyFraction()", float, doc="charged+neutral Electromagnetic Energy Fraction", precision=10),
    )
)

corrT1METJetTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)
jetTable.variables.muonSubtrFactor = Var("1-userFloat('muonSubtrRawPt')/(pt()*jecFactor('Uncorrected'))",float,doc="1-(muon-subtracted raw pt)/(raw pt)",precision=6)

jetForMETTask =  cms.Task(basicJetsForMetForT1METNano,corrT1METJetTable)

#before cross linking
jetUserDataTask = cms.Task(bJetVars,qgtagger,jercVars,pileupJetIdNano)

#before cross linking
jetTask = cms.Task(jetCorrFactorsNano,updatedJets,jetUserDataTask,updatedJetsWithUserData,finalJets)

#after cross linkining
jetTablesTask = cms.Task(bjetNN,cjetNN,jetTable)
