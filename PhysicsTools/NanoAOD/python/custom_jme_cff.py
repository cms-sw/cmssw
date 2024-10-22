import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.simplePATJetFlatTableProducer_cfi import simplePATJetFlatTableProducer

from RecoJets.JetProducers.hfJetShowerShape_cfi import hfJetShowerShape
from RecoJets.JetProducers.PileupJetID_cfi import pileupJetIdCalculator, pileupJetId

from PhysicsTools.NanoAOD.common_cff import Var, P4Vars
from PhysicsTools.NanoAOD.jetsAK4_CHS_cff import jetTable, jetCorrFactorsNano, updatedJets, finalJets, qgtagger
from PhysicsTools.NanoAOD.jetsAK4_Puppi_cff import jetPuppiTable, jetPuppiCorrFactorsNano, updatedJetsPuppi, updatedJetsPuppiWithUserData
from PhysicsTools.NanoAOD.jetMC_cff  import genJetTable, genJetFlavourAssociation, genJetFlavourTable

from PhysicsTools.PatAlgos.tools.jetCollectionTools import GenJetAdder, RecoJetAdder
from PhysicsTools.PatAlgos.tools.jetTools import supportedJetAlgos
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection

bTagCSVV2    = ['pfCombinedInclusiveSecondaryVertexV2BJetTags']
bTagDeepCSV  = ['pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb','pfDeepCSVJetTags:probc','pfDeepCSVJetTags:probudsg']
bTagDeepJet  = [
  'pfDeepFlavourJetTags:probb','pfDeepFlavourJetTags:probbb','pfDeepFlavourJetTags:problepb',
  'pfDeepFlavourJetTags:probc','pfDeepFlavourJetTags:probuds','pfDeepFlavourJetTags:probg'
]
from RecoBTag.ONNXRuntime.pfParticleNetAK4_cff import _pfParticleNetAK4JetTagsAll
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll
from RecoBTag.ONNXRuntime.pfParticleTransformerAK4_cff import _pfParticleTransformerAK4JetTagsAll
from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4_cff import _pfUnifiedParticleTransformerAK4JetTagsAll
bTagDiscriminatorsForAK4 = cms.PSet(foo = cms.vstring(
  bTagDeepJet+
  _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsAll+_pfParticleNetFromMiniAODAK4PuppiForwardJetTagsAll+
  _pfParticleTransformerAK4JetTagsAll + _pfUnifiedParticleTransformerAK4JetTagsAll
))
run2_nanoAOD_ANY.toModify(
  bTagDiscriminatorsForAK4,
  foo = bTagCSVV2+bTagDeepCSV+bTagDeepJet+_pfParticleNetAK4JetTagsAll
)
bTagDiscriminatorsForAK4 = bTagDiscriminatorsForAK4.foo.value()

#
# By default, these collections are saved in NanoAODs:
# - ak4gen (GenJet in NanoAOD), slimmedGenJets in MiniAOD
# - ak8gen (GenJetAK8 in NanoAOD), slimmedGenJetsAK8 in MiniAOD
# Below is a list of genjets that we can save in NanoAOD. Set
# "enabled" to true if you want to store the jet collection
config_genjets = [
  {
    "jet"     : "ak6gen",
    "enabled" : False,
  },
]
config_genjets = list(filter(lambda k: k['enabled'], config_genjets))
#
# GenJets info in NanoAOD
#
nanoInfo_genjets = {
  "ak6gen"  : {
    "name" : "GenJetAK6",
    "doc"  : "AK6 Gen jets (made with visible genparticles) with pt > 3 GeV", # default genjets pt cut after clustering is 3 GeV
  },
}
#
# By default, these collections are saved in the main NanoAODs:
# - ak4pfpuppi  (Jet   in NanoAOD), slimmedJetsPuppi in MiniAOD
# - ak8pfpuppi (FatJet in NanoAOD), slimmedJetsAK8 in MiniAOD
# Below is a list of recojets that we can save in NanoAOD. Set
# "enabled" to true if you want to store the recojet collection.
#
config_recojets = [
  {
    "jet" : "ak4calo",
    "enabled" : True,
    "inputCollection"  : "slimmedCaloJets", #Exist in MiniAOD
    "genJetsCollection": "AK4GenJetsNoNu",
  },
  {
    "jet" : "ak4pf",
    "enabled" : False,
    "inputCollection" : "",
    "genJetsCollection": "AK4GenJetsNoNu",
    "minPtFastjet" : 0.,
  },
  {
    "jet" : "ak8pf",
    "enabled" : False,
    "inputCollection" : "",
    "genJetsCollection": "AK8GenJetsNoNu",
    "minPtFastjet" : 0.,
  },
]
config_recojets = list(filter(lambda k: k['enabled'], config_recojets))
#
# RecoJets info in NanoAOD
#
nanoInfo_recojets = {
  "ak4calo" : {
    "name": "JetCalo",
    "doc" : "AK4 Calo jets (slimmedCaloJets)",
  },
  "ak4pf" : {
    "name"  : "JetPF",
    "doc"   : "AK4 PF jets",
    "ptcut" : "",
  },
  "ak8pf" : {
    "name"  : "FatJetPF",
    "doc"   : "AK8 PF jets",
    "ptcut" : "",
  },
}

GENJETVARS = cms.PSet(P4Vars,
  nConstituents   = jetPuppiTable.variables.nConstituents,
)
PFJETVARS = cms.PSet(P4Vars,
  rawFactor     = jetPuppiTable.variables.rawFactor,
  area          = jetPuppiTable.variables.area,
  chHEF         = jetPuppiTable.variables.chHEF,
  neHEF         = jetPuppiTable.variables.neHEF,
  chEmEF        = jetPuppiTable.variables.chEmEF,
  neEmEF        = jetPuppiTable.variables.neEmEF,
  muEF          = jetPuppiTable.variables.muEF,
  hfHEF         = Var("HFHadronEnergyFraction()",float,doc="hadronic Energy Fraction in HF",precision= 6),
  hfEmEF        = Var("HFEMEnergyFraction()",float,doc="electromagnetic Energy Fraction in HF",precision= 6),
  nMuons        = jetPuppiTable.variables.nMuons,
  nElectrons    = jetPuppiTable.variables.nElectrons,
  nConstituents = jetPuppiTable.variables.nConstituents,
  chHadMultiplicity = Var("chargedHadronMultiplicity()","int16",doc="(Puppi-weighted) number of charged hadrons in the jet"),
  neHadMultiplicity = Var("neutralHadronMultiplicity()","int16",doc="(Puppi-weighted) number of neutral hadrons in the jet"),
  hfHadMultiplicity = Var("HFHadronMultiplicity()", "int16",doc="(Puppi-weighted) number of HF hadrons in the jet"),
  hfEMMultiplicity  = Var("HFEMMultiplicity()","int16",doc="(Puppi-weighted) number of HF EMs in the jet"),
  muMultiplicity    = Var("muonMultiplicity()","int16",doc="(Puppi-weighted) number of muons in the jet"),
  elMultiplicity    = Var("electronMultiplicity()","int16",doc="(Puppi-weighted) number of electrons in the jet"),
  phoMultiplicity   = Var("photonMultiplicity()","int16",doc="(Puppi-weighted) number of photons in the jet"),
  #MC-only variables
  partonFlavour     = Var("partonFlavour()", "int16", doc="flavour from parton matching"),
  hadronFlavour     = Var("hadronFlavour()", "uint8", doc="flavour from hadron ghost clustering"),
  genJetIdx         = Var("?genJetFwdRef().backRef().isNonnull()?genJetFwdRef().backRef().key():-1", "int16", doc="index of matched gen jet"),
)
PUIDVARS = cms.PSet(
  puId_dR2Mean    = Var("?(pt>=10)?userFloat('puId_dR2Mean'):-1",float,doc="pT^2-weighted average square distance of jet constituents from the jet axis (PileUp ID BDT input variable)", precision=14),
  puId_majW       = Var("?(pt>=10)?userFloat('puId_majW'):-1",float,doc="major axis of jet ellipsoid in eta-phi plane (PileUp ID BDT input variable)", precision=14),
  puId_minW       = Var("?(pt>=10)?userFloat('puId_minW'):-1",float,doc="minor axis of jet ellipsoid in eta-phi plane (PileUp ID BDT input variable)", precision=14),
  puId_frac01     = Var("?(pt>=10)?userFloat('puId_frac01'):-1",float,doc="fraction of constituents' pT contained within dR <0.1 (PileUp ID BDT input variable)", precision=14),
  puId_frac02     = Var("?(pt>=10)?userFloat('puId_frac02'):-1",float,doc="fraction of constituents' pT contained within 0.1< dR <0.2 (PileUp ID BDT input variable)", precision=14),
  puId_frac03     = Var("?(pt>=10)?userFloat('puId_frac03'):-1",float,doc="fraction of constituents' pT contained within 0.2< dR <0.3 (PileUp ID BDT input variable)", precision=14),
  puId_frac04     = Var("?(pt>=10)?userFloat('puId_frac04'):-1",float,doc="fraction of constituents' pT contained within 0.3< dR <0.4 (PileUp ID BDT input variable)", precision=14),
  puId_ptD        = Var("?(pt>=10)?userFloat('puId_ptD'):-1",float,doc="pT-weighted average pT of constituents (PileUp ID BDT input variable)", precision=14),
  puId_beta       = Var("?(pt>=10)?userFloat('puId_beta'):-1",float,doc="fraction of pT of charged constituents associated to PV (PileUp ID BDT input variable)", precision=14),
  puId_pull       = Var("?(pt>=10)?userFloat('puId_pull'):-1",float,doc="magnitude of pull vector (PileUp ID BDT input variable)", precision=14),
  puId_jetR       = Var("?(pt>=10)?userFloat('puId_jetR'):-1",float,doc="fraction of jet pT carried by the leading constituent (PileUp ID BDT input variable)", precision=14),
  puId_jetRchg    = Var("?(pt>=10)?userFloat('puId_jetRchg'):-1",float,doc="fraction of jet pT carried by the leading charged constituent (PileUp ID BDT input variable)", precision=14),
  puId_nCharged   = Var("?(pt>=10)?userInt('puId_nCharged'):-1","int16",doc="number of charged constituents (PileUp ID BDT input variable)"),
)
QGLVARS = cms.PSet(
  qgl_axis2       =  Var("?(pt>=10)?userFloat('qgl_axis2'):-1",float,doc="ellipse minor jet axis (Quark vs Gluon likelihood input variable)", precision=14),
  qgl_ptD         =  Var("?(pt>=10)?userFloat('qgl_ptD'):-1",float,doc="pT-weighted average pT of constituents (Quark vs Gluon likelihood input variable)", precision=14),
  qgl_mult        =  Var("?(pt>=10)?userInt('qgl_mult'):-1", "int16",doc="PF candidates multiplicity (Quark vs Gluon likelihood input variable)"),
)
BTAGVARS = cms.PSet(
  btagDeepB = Var("?(pt>=15)&&((bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'))>=0)?bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'):-1",float,doc="DeepCSV b+bb tag discriminator",precision=10),
  btagDeepCvL = Var("?(pt>=15)&&(bDiscriminator('pfDeepCSVJetTags:probc')>=0)?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probudsg')):-1", float,doc="DeepCSV c vs udsg discriminator",precision=10),
  btagDeepCvB = Var("?(pt>=15)&&bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')):-1",float,doc="DeepCSV c vs b+bb discriminator",precision=10),
)
DEEPJETVARS = cms.PSet(
  btagDeepFlavB   = Var("?(pt>=15)?bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'):-1",float,doc="DeepJet b+bb+lepb tag discriminator",precision=10),
  btagDeepFlavC   = Var("?(pt>=15)?bDiscriminator('pfDeepFlavourJetTags:probc'):-1",float,doc="DeepFlavour charm tag raw score",precision=10),
  btagDeepFlavG   = Var("?(pt>=15)?bDiscriminator('pfDeepFlavourJetTags:probg'):-1",float,doc="DeepFlavour gluon tag raw score",precision=10),
  btagDeepFlavUDS = Var("?(pt>=15)?bDiscriminator('pfDeepFlavourJetTags:probuds'):-1",float,doc="DeepFlavour uds tag raw score",precision=10),
  btagDeepFlavCvL = Var("?(pt>=15)&&(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probuds')+bDiscriminator('pfDeepFlavourJetTags:probg')):-1",float,doc="DeepJet c vs uds+g discriminator",precision=10),
  btagDeepFlavCvB = Var("?(pt>=15)&&(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb'))>0?bDiscriminator('pfDeepFlavourJetTags:probc')/(bDiscriminator('pfDeepFlavourJetTags:probc')+bDiscriminator('pfDeepFlavourJetTags:probb')+bDiscriminator('pfDeepFlavourJetTags:probbb')+bDiscriminator('pfDeepFlavourJetTags:problepb')):-1",float,doc="DeepJet c vs b+bb+lepb discriminator",precision=10),
  btagDeepFlavQG  = Var("?(pt>=15)&&(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds'))>0?bDiscriminator('pfDeepFlavourJetTags:probg')/(bDiscriminator('pfDeepFlavourJetTags:probg')+bDiscriminator('pfDeepFlavourJetTags:probuds')):-1",float,doc="DeepJet g vs uds discriminator",precision=10),
)
ROBUSTPARTAK4VARS = cms.PSet(
  btagRobustParTAK4B   = Var("?(pt>=15)?bDiscriminator('pfParticleTransformerAK4JetTags:probb')+bDiscriminator('pfParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfParticleTransformerAK4JetTags:problepb'):-1",float,doc="RobustParTAK4 b+bb+lepb tag discriminator",precision=10),
  btagRobustParTAK4C   = Var("?(pt>=15)?bDiscriminator('pfParticleTransformerAK4JetTags:probc'):-1",float,doc="RobustParTAK4 charm tag raw score",precision=10),
  btagRobustParTAK4G   = Var("?(pt>=15)?bDiscriminator('pfParticleTransformerAK4JetTags:probg'):-1",float,doc="RobustParTAK4 gluon tag raw score",precision=10),
  btagRobustParTAK4UDS = Var("?(pt>=15)?bDiscriminator('pfParticleTransformerAK4JetTags:probuds'):-1",float,doc="RobustParTAK4 uds tag raw score",precision=10),
  btagRobustParTAK4CvL = Var("?(pt>=15)&&(bDiscriminator('pfParticleTransformerAK4JetTags:probc')+bDiscriminator('pfParticleTransformerAK4JetTags:probuds')+bDiscriminator('pfParticleTransformerAK4JetTags:probg'))>0?bDiscriminator('pfParticleTransformerAK4JetTags:probc')/(bDiscriminator('pfParticleTransformerAK4JetTags:probc')+bDiscriminator('pfParticleTransformerAK4JetTags:probuds')+bDiscriminator('pfParticleTransformerAK4JetTags:probg')):-1",float,doc="RobustParTAK4 c vs uds+g discriminator",precision=10),
  btagRobustParTAK4CvB = Var("?(pt>=15)&&(bDiscriminator('pfParticleTransformerAK4JetTags:probc')+bDiscriminator('pfParticleTransformerAK4JetTags:probb')+bDiscriminator('pfParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfParticleTransformerAK4JetTags:problepb'))>0?bDiscriminator('pfParticleTransformerAK4JetTags:probc')/(bDiscriminator('pfParticleTransformerAK4JetTags:probc')+bDiscriminator('pfParticleTransformerAK4JetTags:probb')+bDiscriminator('pfParticleTransformerAK4JetTags:probbb')+bDiscriminator('pfParticleTransformerAK4JetTags:problepb')):-1",float,doc="RobustParTAK4 c vs b+bb+lepb discriminator",precision=10),
  btagRobustParTAK4QG  = Var("?(pt>=15)&&(bDiscriminator('pfParticleTransformerAK4JetTags:probg')+bDiscriminator('pfParticleTransformerAK4JetTags:probuds'))>0?bDiscriminator('pfParticleTransformerAK4JetTags:probg')/(bDiscriminator('pfParticleTransformerAK4JetTags:probg')+bDiscriminator('pfParticleTransformerAK4JetTags:probuds')):-1",float,doc="RobustParTAK4 g vs uds discriminator",precision=10),
)
UNIFIEDPARTAK4VARS = cms.PSet(
  btagUParTAK4B = Var("?pt>15 && bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll'):-1",float,precision=10,doc="UnifiedParTAK4 b vs. udscg"),
  btagUParTAK4CvL = Var("?pt>15 && bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsL'):-1",float,precision=10,doc="UnifiedParTAK4 c vs. udsg"),
  btagUParTAK4CvB = Var("?pt>15 && bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:CvsB'):-1",float,precision=10,doc="UnifiedParTAK4 c vs. b"),
  btagUParTAK4SvCB  = Var("?pt>15 && bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsBC')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsBC'):-1",float,precision=10,doc="UnifiedParTAK4 s vs. bc"),
  btagUParTAK4SvUDG = Var("?pt>15 && bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsUDG')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:SvsUDG'):-1",float,precision=10,doc="UnifiedParTAK4 s vs. udg"),
  btagUParTAK4UDG = Var("?pt>15? bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probu')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probd')+bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:probg'):-1",float,precision=10,doc="UnifiedParTAK4 u+d+g raw score"),
  btagUParTAK4QvG = Var("?pt>15 && bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:QvsG'):-1",float,precision=10,doc="UnifiedParTAK4 q (uds) vs. g"),
  btagUParTAK4TauVJet = Var("?pt>15 && bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:TauVsJet')>0?bDiscriminator('pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:TauVsJet'):-1",float,precision=10,doc="UnifiedParTAK4 tau vs. jet"),
  UParTAK4RegPtRawCorr = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptcorr'):-1",float,precision=10,doc="UnifiedParTAK4 universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT"),
  UParTAK4RegPtRawCorrNeutrino = Var("?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu')>0?bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptnu'):-1",float,precision=10,doc="UnifiedParTAK4 universal flavor-aware pT regression neutrino correction, relative to visible. To apply full regression, multiply raw jet pT by both UParTAK4RegPtRawCorr and UParTAK4RegPtRawCorrNeutrino."),
  UParTAK4RegPtRawRes = Var("?0.5*(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow')) > 0?0.5*(bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreshigh')-bDiscriminator('pfUnifiedParticleTransformerAK4JetTags:ptreslow')):-1",float,precision=10,doc="UnifiedParTAK4 universal flavor-aware jet pT resolution estimator, (q84 - q16)/2"),
)
PARTICLENETAK4VARS = cms.PSet(
  particleNetAK4_B = Var("?(pt>=15)?bDiscriminator('pfParticleNetAK4DiscriminatorsJetTags:BvsAll'):-1",float,doc="ParticleNetAK4 tagger b vs all (udsg, c) discriminator",precision=10),
  particleNetAK4_CvsL = Var("?(pt>=15)?bDiscriminator('pfParticleNetAK4DiscriminatorsJetTags:CvsL'):-1",float,doc="ParticleNetAK4 tagger c vs udsg discriminator",precision=10),
  particleNetAK4_CvsB = Var("?(pt>=15)?bDiscriminator('pfParticleNetAK4DiscriminatorsJetTags:CvsB'):-1",float,doc="ParticleNetAK4 tagger c vs b discriminator",precision=10),
  particleNetAK4_QvsG = Var("?(pt>=15)?bDiscriminator('pfParticleNetAK4DiscriminatorsJetTags:QvsG'):-1",float,doc="ParticleNetAK4 tagger uds vs g discriminator",precision=10),
  particleNetAK4_G = Var("?(pt>=15)?bDiscriminator('pfParticleNetAK4JetTags:probg'):-1",float,doc="ParticleNetAK4 tagger g raw score",precision=10),
  particleNetAK4_puIdDisc = Var("?(pt>=15)?1-bDiscriminator('pfParticleNetAK4JetTags:probpu'):-1",float,doc="ParticleNetAK4 tagger pileup jet discriminator",precision=10),
)

CALOJETVARS = cms.PSet(P4Vars,
  area      = jetPuppiTable.variables.area,
  rawFactor = jetPuppiTable.variables.rawFactor,
  emf       = Var("emEnergyFraction()", float, doc = "electromagnetic energy fraction", precision = 10),
)


#******************************************
#
#
# Reco Jets related functions
#
#
#******************************************
def AddPileUpJetIDVars(proc, jetName="", jetSrc="", jetTableName="", jetTaskName=""):
  """
  Setup modules to calculate pileup jet ID input variables for PF jet
  """

  #
  # Calculate pileup jet ID variables
  #
  puJetIdVarsCalculator = "puJetIdCalculator{}".format(jetName)
  setattr(proc, puJetIdVarsCalculator, pileupJetIdCalculator.clone(
      jets = jetSrc,
      vertexes  = "offlineSlimmedPrimaryVertices",
      inputIsCorrected = True,
      applyJec  = False,
      srcConstituentWeights = "packedpuppi" if "PUPPI" in jetName.upper() else ""
    )
  )
  getattr(proc,jetTaskName).add(getattr(proc, puJetIdVarsCalculator))

  #
  # Get the variables
  #
  puJetIDVar = "puJetIDVar{}".format(jetName)
  setattr(proc, puJetIDVar, cms.EDProducer("PileupJetIDVarProducer",
      srcJet = cms.InputTag(jetSrc),
      srcPileupJetId = cms.InputTag(puJetIdVarsCalculator)
    )
  )
  getattr(proc,jetTaskName).add(getattr(proc, puJetIDVar))

  #
  # Save variables as userFloats and userInts for each jet
  #
  patJetWithUserData = "{}WithUserData".format(jetSrc)
  getattr(proc,patJetWithUserData).userFloats.puId_dR2Mean  = cms.InputTag("{}:dR2Mean".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_majW     = cms.InputTag("{}:majW".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_minW     = cms.InputTag("{}:minW".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac01   = cms.InputTag("{}:frac01".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac02   = cms.InputTag("{}:frac02".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac03   = cms.InputTag("{}:frac03".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_frac04   = cms.InputTag("{}:frac04".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_ptD      = cms.InputTag("{}:ptD".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_beta     = cms.InputTag("{}:beta".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_pull     = cms.InputTag("{}:pull".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_jetR     = cms.InputTag("{}:jetR".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userFloats.puId_jetRchg  = cms.InputTag("{}:jetRchg".format(puJetIDVar))
  getattr(proc,patJetWithUserData).userInts.puId_nCharged   = cms.InputTag("{}:nCharged".format(puJetIDVar))

  #
  # Specfiy variables in the jet table to save in NanoAOD
  #
  getattr(proc,jetTableName).variables.puId_dR2Mean  = PUIDVARS.puId_dR2Mean
  getattr(proc,jetTableName).variables.puId_majW     = PUIDVARS.puId_majW
  getattr(proc,jetTableName).variables.puId_minW     = PUIDVARS.puId_minW
  getattr(proc,jetTableName).variables.puId_frac01   = PUIDVARS.puId_frac01
  getattr(proc,jetTableName).variables.puId_frac02   = PUIDVARS.puId_frac02
  getattr(proc,jetTableName).variables.puId_frac03   = PUIDVARS.puId_frac03
  getattr(proc,jetTableName).variables.puId_frac04   = PUIDVARS.puId_frac04
  getattr(proc,jetTableName).variables.puId_ptD      = PUIDVARS.puId_ptD
  getattr(proc,jetTableName).variables.puId_beta     = PUIDVARS.puId_beta
  getattr(proc,jetTableName).variables.puId_pull     = PUIDVARS.puId_pull
  getattr(proc,jetTableName).variables.puId_jetR     = PUIDVARS.puId_jetR
  getattr(proc,jetTableName).variables.puId_jetRchg  = PUIDVARS.puId_jetRchg
  getattr(proc,jetTableName).variables.puId_nCharged = PUIDVARS.puId_nCharged

  return proc

def AddQGLTaggerVars(proc, jetName="", jetSrc="", jetTableName="", jetTaskName="", calculateQGLVars=False):
  """
  Schedule the QGTagger module to calculate input variables to the QG likelihood
  """

  isPUPPIJet = True if "PUPPI" in jetName.upper() else False

  QGLTagger="qgtagger{}".format(jetName)
  patJetWithUserData="{}WithUserData".format(jetSrc)

  if calculateQGLVars:
    setattr(proc, QGLTagger, qgtagger.clone(
        srcJets = jetSrc,
        computeLikelihood = False,
      )
    )
    if isPUPPIJet:
      getattr(proc,QGLTagger).srcConstituentWeights = cms.InputTag("packedpuppi")

  #
  # Save variables as userFloats and userInts for each jet
  #
  getattr(proc,patJetWithUserData).userFloats.qgl_axis2 = cms.InputTag(QGLTagger+":axis2")
  getattr(proc,patJetWithUserData).userFloats.qgl_ptD   = cms.InputTag(QGLTagger+":ptD")
  getattr(proc,patJetWithUserData).userInts.qgl_mult    = cms.InputTag(QGLTagger+":mult")

  #
  # Specfiy variables in the jet table to save in NanoAOD
  #
  getattr(proc,jetTableName).variables.qgl_axis2 =  QGLVARS.qgl_axis2
  getattr(proc,jetTableName).variables.qgl_ptD   =  QGLVARS.qgl_ptD
  getattr(proc,jetTableName).variables.qgl_mult  =  QGLVARS.qgl_mult

  if calculateQGLVars:
    getattr(proc,jetTaskName).add(getattr(proc, QGLTagger))

  return proc

def AddBTaggingScores(proc, jetTableName=""):
  """
  Store b-tagging scores from various algortihm
  """

  getattr(proc, jetTableName).variables.btagDeepFlavB   = DEEPJETVARS.btagDeepFlavB
  getattr(proc, jetTableName).variables.btagDeepFlavCvL = DEEPJETVARS.btagDeepFlavCvL
  getattr(proc, jetTableName).variables.btagDeepFlavCvB = DEEPJETVARS.btagDeepFlavCvB

  run2_nanoAOD_ANY.toModify(
    getattr(proc, jetTableName).variables,
    btagCSVV2 = Var("bDiscriminator('pfCombinedInclusiveSecondaryVertexV2BJetTags')",float,doc=" pfCombinedInclusiveSecondaryVertexV2 b-tag discriminator (aka CSVV2)",precision=10),
    btagDeepB = Var("?(bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'))>=0?bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb'):-1",float,doc="DeepCSV b+bb tag discriminator",precision=10),
    btagDeepCvL = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probudsg')):-1", float,doc="DeepCSV c vs udsg discriminator",precision=10),
    btagDeepCvB = Var("?bDiscriminator('pfDeepCSVJetTags:probc')>=0?bDiscriminator('pfDeepCSVJetTags:probc')/(bDiscriminator('pfDeepCSVJetTags:probc')+bDiscriminator('pfDeepCSVJetTags:probb')+bDiscriminator('pfDeepCSVJetTags:probbb')):-1",float,doc="DeepCSV c vs b+bb discriminator",precision=10)
  )

  return proc

def AddDeepJetGluonLQuarkScores(proc, jetTableName=""):
  """
  Store DeepJet raw score in jetTable for gluon and light quark
  """

  getattr(proc, jetTableName).variables.btagDeepFlavG   = DEEPJETVARS.btagDeepFlavG
  getattr(proc, jetTableName).variables.btagDeepFlavUDS = DEEPJETVARS.btagDeepFlavUDS
  getattr(proc, jetTableName).variables.btagDeepFlavQG  = DEEPJETVARS.btagDeepFlavQG

  return proc

def AddRobustParTAK4Scores(proc, jetTableName=""):
  """
  Store RobustParTAK4 scores in jetTable
  """

  getattr(proc, jetTableName).variables.btagRobustParTAK4B = ROBUSTPARTAK4VARS.btagRobustParTAK4B
  getattr(proc, jetTableName).variables.btagRobustParTAK4CvL = ROBUSTPARTAK4VARS.btagRobustParTAK4CvL
  getattr(proc, jetTableName).variables.btagRobustParTAK4CvB = ROBUSTPARTAK4VARS.btagRobustParTAK4CvB

  return proc

def AddUnifiedParTAK4Scores(proc, jetTableName=""):
  """
  Store RobustParTAK4 scores in jetTable
  """

  getattr(proc, jetTableName).variables.btagUParTAK4B = UNIFIEDPARTAK4VARS.btagUParTAK4B
  getattr(proc, jetTableName).variables.btagUParTAK4CvL = UNIFIEDPARTAK4VARS.btagUParTAK4CvL
  getattr(proc, jetTableName).variables.btagUParTAK4CvB = UNIFIEDPARTAK4VARS.btagUParTAK4CvB
  getattr(proc, jetTableName).variables.btagUParTAK4SvCB = UNIFIEDPARTAK4VARS.btagUParTAK4SvCB
  getattr(proc, jetTableName).variables.btagUParTAK4SvUDG = UNIFIEDPARTAK4VARS.btagUParTAK4SvUDG
  getattr(proc, jetTableName).variables.btagUParTAK4UDG = UNIFIEDPARTAK4VARS.btagUParTAK4UDG
  getattr(proc, jetTableName).variables.btagUParTAK4QvG = UNIFIEDPARTAK4VARS.btagUParTAK4QvG
  getattr(proc, jetTableName).variables.btagUParTAK4TauVJet = UNIFIEDPARTAK4VARS.btagUParTAK4TauVJet
  getattr(proc, jetTableName).variables.UParTAK4RegPtRawCorr = UNIFIEDPARTAK4VARS.UParTAK4RegPtRawCorr
  getattr(proc, jetTableName).variables.UParTAK4RegPtRawCorrNeutrino = UNIFIEDPARTAK4VARS.UParTAK4RegPtRawCorrNeutrino
  getattr(proc, jetTableName).variables.UParTAK4RegPtRawRes = UNIFIEDPARTAK4VARS.UParTAK4RegPtRawRes

  return proc

def AddParticleNetAK4Scores(proc, jetTableName=""):
  """
  Store ParticleNetAK4 scores in jetTable
  """

  getattr(proc, jetTableName).variables.particleNetAK4_B = PARTICLENETAK4VARS.particleNetAK4_B
  getattr(proc, jetTableName).variables.particleNetAK4_CvsL = PARTICLENETAK4VARS.particleNetAK4_CvsL
  getattr(proc, jetTableName).variables.particleNetAK4_CvsB = PARTICLENETAK4VARS.particleNetAK4_CvsB
  getattr(proc, jetTableName).variables.particleNetAK4_QvsG = PARTICLENETAK4VARS.particleNetAK4_QvsG
  getattr(proc, jetTableName).variables.particleNetAK4_G = PARTICLENETAK4VARS.particleNetAK4_G
  getattr(proc, jetTableName).variables.particleNetAK4_puIdDisc = PARTICLENETAK4VARS.particleNetAK4_puIdDisc

  return proc

def AddNewPatJets(proc, recoJetInfo, runOnMC):
  """
  Add patJet into custom nanoAOD
  """

  jetName = recoJetInfo.jetUpper
  payload = recoJetInfo.jetCorrPayload
  doPF    = recoJetInfo.doPF
  doCalo  = recoJetInfo.doCalo
  patJetFinalColl = recoJetInfo.patJetFinalCollection

  nanoInfoForJet = nanoInfo_recojets[recoJetInfo.jet]
  jetTablePrefix = nanoInfoForJet["name"]
  jetTableDoc    = nanoInfoForJet["doc"]
  ptcut          = nanoInfoForJet["ptcut"] if "ptcut" in nanoInfoForJet else 8
  doPUIDVar      = nanoInfoForJet["doPUIDVar"] if "doPUIDVar" in nanoInfoForJet else False
  doQGL          = nanoInfoForJet["doQGL"] if "doQGL" in nanoInfoForJet else False
  doBTag         = nanoInfoForJet["doBTag"] if "doBTag" in nanoInfoForJet else False

  SavePatJets(proc,
    jetName, payload, patJetFinalColl, jetTablePrefix, jetTableDoc, doPF, doCalo,
    ptcut=ptcut, doPUIDVar=doPUIDVar, doQGL=doQGL, doBTag=doBTag, runOnMC=runOnMC
  )

  return proc

def SavePatJets(proc, jetName, payload, patJetFinalColl, jetTablePrefix, jetTableDoc,
                doPF, doCalo, ptcut, doPUIDVar=False, doQGL=False, doBTag=False, runOnMC=False):
  """
  Schedule modules for a given patJet collection and save its variables into custom NanoAOD
  """

  #
  # Setup jet correction factors
  #
  jetCorrFactors = "jetCorrFactorsNano{}".format(jetName)
  setattr(proc, jetCorrFactors, jetCorrFactorsNano.clone(
      src = patJetFinalColl,
      payload = payload,
    )
  )

  #
  # Update jets
  #
  srcJets = "updatedJets{}".format(jetName)
  setattr(proc, srcJets, updatedJets.clone(
      jetSource = patJetFinalColl,
      jetCorrFactorsSource = [jetCorrFactors],
    )
  )

  #
  # Setup UserDataEmbedder
  #
  srcJetsWithUserData = "updatedJets{}WithUserData".format(jetName)
  setattr(proc, srcJetsWithUserData, cms.EDProducer("PATJetUserDataEmbedder",
      src = cms.InputTag(srcJets),
      userFloats = cms.PSet(),
      userInts = cms.PSet(),
    )
  )

  #
  # Filter jets with pt cut
  #
  finalJetsCut = "(pt >= {ptcut:.0f})".format(ptcut=ptcut)
  if runOnMC:
    finalJetsCut = "(pt >= {ptcut:.0f}) || ((pt < {ptcut:.0f}) && (genJetFwdRef().backRef().isNonnull()))".format(ptcut=ptcut)

  finalJetsForTable = "finalJets{}".format(jetName)
  setattr(proc, finalJetsForTable, finalJets.clone(
      src = srcJetsWithUserData,
      cut = finalJetsCut
    )
  )

  #
  # Save jets in table
  #
  tableContent = PFJETVARS
  if doCalo:
    tableContent = CALOJETVARS

  jetTableCutDefault = "" #Don't apply any cuts for the table.

  jetTableDocDefault = jetTableDoc + " with JECs applied. Jets with pt >= {ptcut:.0f} GeV are stored.".format(ptcut=ptcut)
  if runOnMC:
    jetTableDocDefault += "For jets with pt < {ptcut:.0f} GeV, only those matched to gen jets are stored.".format(ptcut=ptcut)

  if doCalo:
    jetTableDocDefault = jetTableDoc

  jetTableName = "jet{}Table".format(jetName)
  setattr(proc,jetTableName, simplePATJetFlatTableProducer.clone(
      src = cms.InputTag(finalJetsForTable),
      cut = cms.string(jetTableCutDefault),
      name = cms.string(jetTablePrefix),
      doc  = cms.string(jetTableDocDefault),
      variables = cms.PSet(tableContent)
    )
  )
  getattr(proc,jetTableName).variables.pt.precision=10
  getattr(proc,jetTableName).variables.rawFactor.precision=10

  #
  # Save MC-only jet variables in table
  #
  jetMCTableName = "jet{}MCTable".format(jetName)
  setattr(proc, jetMCTableName, simplePATJetFlatTableProducer.clone(
      src = cms.InputTag(finalJetsForTable),
      cut = getattr(proc,jetTableName).cut,
      name = cms.string(jetTablePrefix),
      extension = cms.bool(True), # this is an extension table
      variables = cms.PSet(
      partonFlavour = PFJETVARS.partonFlavour,
      hadronFlavour = PFJETVARS.hadronFlavour,
      genJetIdx = PFJETVARS.genJetIdx,
      )
    )
  )

  #
  # Define the jet modules Task first
  #
  jetTaskName = "jet{}Task".format(jetName)
  setattr(proc, jetTaskName, cms.Task(
     getattr(proc,jetCorrFactors),
     getattr(proc,srcJets),
     getattr(proc,srcJetsWithUserData),
     getattr(proc,finalJetsForTable)
   )
  )
  proc.nanoTableTaskCommon.add(getattr(proc,jetTaskName))

  #
  # Define the jet tables Task
  #
  jetTableTaskName = "jet{}TablesTask".format(jetName)
  setattr(proc, jetTableTaskName, cms.Task(getattr(proc,jetTableName)))
  proc.nanoTableTaskCommon.add(getattr(proc,jetTableTaskName))

  jetMCTableTaskName = "jet{}MCTablesTask".format(jetName)
  setattr(proc, jetMCTableTaskName, cms.Task(getattr(proc,jetMCTableName)))
  if runOnMC:
    proc.nanoTableTaskFS.add(getattr(proc,jetMCTableTaskName))

  #
  # Schedule plugins to calculate Jet ID, PileUp Jet ID input variables, and Quark-Gluon Likehood input variables.
  #
  if doPF:
    if doPUIDVar:
      proc = AddPileUpJetIDVars(proc, jetName=jetName, jetSrc=srcJets, jetTableName=jetTableName, jetTaskName=jetTaskName)
    if doQGL:
      proc = AddQGLTaggerVars(proc,jetName=jetName, jetSrc=srcJets, jetTableName=jetTableName, jetTaskName=jetTaskName, calculateQGLVars=True)

  #
  # Save b-tagging algorithm scores. Should only be done for jet collection with b-tagging
  # calculated when reclustered or collection saved with b-tagging info in MiniAOD
  #
  if doBTag:
    AddBTaggingScores(proc,jetTableName=jetTableName)
    AddDeepJetGluonLQuarkScores(proc,jetTableName=jetTableName)
    AddParticleNetAK4Scores(proc,jetTableName=jetTableName)
    AddRobustParTAK4Scores(proc,jetTableName=jetTableName)
    AddUnifiedParTAK4Scores(proc,jetTableName=jetTableName)

  return proc


def ReclusterAK4PuppiJets(proc, recoJA, runOnMC):
  """
  Recluster AK4 Puppi jets and replace slimmedJetsPuppi
  that is used as default to save AK4 Puppi jets in NanoAODs.
  """
  print("custom_jme_cff::ReclusterAK4PuppiJets: Recluster AK4 PF Puppi jets")

  #
  # Recluster AK4 Puppi jets
  #
  cfg = {
    "jet" : "ak4pfpuppi",
    "inputCollection" : "",
    "genJetsCollection": "AK4GenJetsNoNu",
    "bTagDiscriminators": bTagDiscriminatorsForAK4,
    "minPtFastjet" : 0.,
  }
  recoJetInfo = recoJA.addRecoJetCollection(proc, **cfg)

  jetName = recoJetInfo.jetUpper
  patJetFinalColl = recoJetInfo.patJetFinalCollection

  #
  # Change the input jet source for jetCorrFactorsNano
  # and updatedJets
  #
  proc.jetPuppiCorrFactorsNano.src=patJetFinalColl
  proc.updatedJetsPuppi.jetSource=patJetFinalColl

  #
  # Change pt cut
  #
  finalJetsPuppiCut = ""
  if runOnMC:
    finalJetsPuppiCut = "(pt >= 8) || ((pt < 8) && (genJetFwdRef().backRef().isNonnull()))"
  else:
    finalJetsPuppiCut = "(pt >= 8)"

  proc.finalJetsPuppi.cut = finalJetsPuppiCut
  #
  # Add a minimum pt cut for corrT1METJets.
  #
  proc.corrT1METJetPuppiTable.cut = "pt>=8 && pt<15 && abs(eta)<9.9"

  #
  # Jet table
  #
  # For Run-2 eras, the main AK4 jet collection in NanoAOD is the CHS collection
  run2_nanoAOD_ANY.toModify(
    proc.jetTable, name = "Jet"
  ).toModify(
    # So need to change the table name for AK4 puppi here
    proc.jetPuppiTable,
    name = "JetPuppi",
    src = cms.InputTag("finalJetsPuppi")
  )

  #
  # Jet table documentation
  #
  jetPuppiTableDoc = "AK4 PF Puppi jets with JECs applied. Jets with pt >= 8 GeV are stored."
  if runOnMC:
    jetPuppiTableDoc += "For jets with pt < 8 GeV, only those matched to AK4 Gen jets are stored."
  proc.jetPuppiTable.doc = jetPuppiTableDoc

  proc.jetPuppiTable.variables.rawFactor.precision = 10

  #
  # Add variables
  #
  proc.jetPuppiTable.variables.chHadMultiplicity = PFJETVARS.chHadMultiplicity
  proc.jetPuppiTable.variables.neHadMultiplicity = PFJETVARS.neHadMultiplicity
  proc.jetPuppiTable.variables.hfHadMultiplicity = PFJETVARS.hfHadMultiplicity
  proc.jetPuppiTable.variables.hfEMMultiplicity  = PFJETVARS.hfEMMultiplicity
  proc.jetPuppiTable.variables.muMultiplicity    = PFJETVARS.muMultiplicity
  proc.jetPuppiTable.variables.elMultiplicity    = PFJETVARS.elMultiplicity
  proc.jetPuppiTable.variables.phoMultiplicity   = PFJETVARS.phoMultiplicity

  #
  #
  #

  from PhysicsTools.NanoAOD.patJetPFConstituentVarProducer_cfi import patJetPFConstituentVarProducer
  jetPFConstituentVarName = "jetPFConstituentVar{}".format(jetName)
  setattr(proc, jetPFConstituentVarName, patJetPFConstituentVarProducer.clone(
      jets = "updatedJetsPuppi",
      puppi_value_map = 'packedpuppi',
      fallback_puppi_weight = False,
    )
  )
  proc.jetPuppiTask.add(getattr(proc, jetPFConstituentVarName))
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstNeHadEF = cms.InputTag(jetPFConstituentVarName+':leadConstNeHadEF')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstChHadEF = cms.InputTag(jetPFConstituentVarName+':leadConstChHadEF')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstPhotonEF = cms.InputTag(jetPFConstituentVarName+':leadConstPhotonEF')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstElectronEF = cms.InputTag(jetPFConstituentVarName+':leadConstElectronEF')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstMuonEF = cms.InputTag(jetPFConstituentVarName+':leadConstMuonEF')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstHFHADEF = cms.InputTag(jetPFConstituentVarName+':leadConstHFHADEF')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstHFEMEF = cms.InputTag(jetPFConstituentVarName+':leadConstHFEMEF')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstNeHadPuppiWeight = cms.InputTag(jetPFConstituentVarName+':leadConstNeHadPuppiWeight')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstChHadPuppiWeight = cms.InputTag(jetPFConstituentVarName+':leadConstChHadPuppiWeight')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstPhotonPuppiWeight = cms.InputTag(jetPFConstituentVarName+':leadConstPhotonPuppiWeight')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstElectronPuppiWeight = cms.InputTag(jetPFConstituentVarName+':leadConstElectronPuppiWeight')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstMuonPuppiWeight = cms.InputTag(jetPFConstituentVarName+':leadConstMuonPuppiWeight')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstHFHADPuppiWeight = cms.InputTag(jetPFConstituentVarName+':leadConstHFHADPuppiWeight')
  proc.updatedJetsPuppiWithUserData.userFloats.leadConstHFEMPuppiWeight = cms.InputTag(jetPFConstituentVarName+':leadConstHFEMPuppiWeight')
  proc.jetPuppiTable.variables.leadConstNeHadEF = Var("userFloat('leadConstNeHadEF')",float,doc="Leading PF neutral hadron constituent energy fraction w.r.t jet raw energy",precision=10)
  proc.jetPuppiTable.variables.leadConstChHadEF = Var("userFloat('leadConstChHadEF')",float,doc="Leading PF charged hadron constituent energy fraction w.r.t jet raw energy",precision=10)
  proc.jetPuppiTable.variables.leadConstPhotonEF = Var("userFloat('leadConstPhotonEF')",float,doc="Leading PF photon constituent energy fraction w.r.t jet raw energy",precision=10)
  proc.jetPuppiTable.variables.leadConstElectronEF = Var("userFloat('leadConstElectronEF')",float,doc="Leading PF electron constituent energy fraction w.r.t jet raw energy",precision=10)
  proc.jetPuppiTable.variables.leadConstMuonEF = Var("userFloat('leadConstMuonEF')",float,doc="Leading PF muon constituent energy fraction w.r.t jet raw energy",precision=10)
  proc.jetPuppiTable.variables.leadConstHFHADEF = Var("userFloat('leadConstHFHADEF')",float,doc="Leading PF HF HAD constituent energy fraction w.r.t jet raw energy",precision=10)
  proc.jetPuppiTable.variables.leadConstHFEMEF = Var("userFloat('leadConstHFEMEF')",float,doc="Leading PF HF EM constituent energy fraction w.r.t jet raw energy",precision=10)
  proc.jetPuppiTable.variables.leadConstNeHadPuppiWeight = Var("userFloat('leadConstNeHadPuppiWeight')",float,doc="Leading PF neutral hadron constituent puppi weight",precision=10)
  proc.jetPuppiTable.variables.leadConstChHadPuppiWeight = Var("userFloat('leadConstChHadPuppiWeight')",float,doc="Leading PF charged hadron constituent puppi weight",precision=10)
  proc.jetPuppiTable.variables.leadConstPhotonPuppiWeight = Var("userFloat('leadConstPhotonPuppiWeight')",float,doc="Leading PF photon constituent puppi weight",precision=10)
  proc.jetPuppiTable.variables.leadConstElectronPuppiWeight = Var("userFloat('leadConstElectronPuppiWeight')",float,doc="Leading PF electron constituent puppi weight",precision=10)
  proc.jetPuppiTable.variables.leadConstMuonPuppiWeight = Var("userFloat('leadConstMuonPuppiWeight')",float,doc="Leading PF muon constituent puppi weight",precision=10)
  proc.jetPuppiTable.variables.leadConstHFHADPuppiWeight = Var("userFloat('leadConstHFHADPuppiWeight')",float,doc="Leading PF HF HAD constituent puppi weight",precision=10)
  proc.jetPuppiTable.variables.leadConstHFEMPuppiWeight = Var("userFloat('leadConstHFEMPuppiWeight')",float,doc="Leading PF HF EM constituent puppi weight",precision=10)


  #
  # Add Pileup Jet ID for Puppi jets
  #
  from RecoJets.JetProducers.PileupJetID_cfi import pileupJetIdPuppi
  pileupJetIdName = "pileupJetId{}".format(jetName)
  setattr(proc, pileupJetIdName, pileupJetIdPuppi.clone(
      jets = "updatedJetsPuppi",
      srcConstituentWeights = "packedpuppi",
      vertexes  = "offlineSlimmedPrimaryVertices",
      inputIsCorrected=True,
      applyJec=False
    )
  )
  proc.jetPuppiTask.add(getattr(proc, pileupJetIdName))
  proc.updatedJetsPuppiWithUserData.userFloats.puIdDisc = cms.InputTag(pileupJetIdName+':fullDiscriminant')
  proc.jetPuppiTable.variables.puIdDisc = Var("userFloat('puIdDisc')", float, doc="Pileup ID BDT discriminant with 133X Winter24 PuppiV18 training",precision=10)


  #
  # Add variables for pileup jet ID studies.
  #
  proc = AddPileUpJetIDVars(proc,
    jetName = jetName,
    jetSrc = "updatedJetsPuppi",
    jetTableName = "jetPuppiTable",
    jetTaskName = "jetPuppiTask"
  )
  #
  # Add variables for quark guon likelihood tagger studies.
  # Save variables as userFloats and userInts in each jet
  #
  proc = AddQGLTaggerVars(proc,
                          jetName = jetName,
                          jetSrc = "updatedJetsPuppi",
                          jetTableName = "jetPuppiTable",
                          jetTaskName = "jetPuppiTask",
                          calculateQGLVars=True
                        )
  #
  # Save DeepJet b-tagging and c-tagging variables
  #
  proc.jetPuppiTable.variables.btagDeepFlavB   = DEEPJETVARS.btagDeepFlavB
  proc.jetPuppiTable.variables.btagDeepFlavCvL = DEEPJETVARS.btagDeepFlavCvL
  proc.jetPuppiTable.variables.btagDeepFlavCvB = DEEPJETVARS.btagDeepFlavCvB
  #
  # Save DeepJet raw score for gluon and light quarks
  #
  proc.jetPuppiTable.variables.btagDeepFlavG   = DEEPJETVARS.btagDeepFlavG
  proc.jetPuppiTable.variables.btagDeepFlavUDS = DEEPJETVARS.btagDeepFlavUDS
  proc.jetPuppiTable.variables.btagDeepFlavQG  = DEEPJETVARS.btagDeepFlavQG
  #
  # Save RobustParTAK4 b-tagging and c-tagging variables
  #
  proc.jetPuppiTable.variables.btagRobustParTAK4B   = ROBUSTPARTAK4VARS.btagRobustParTAK4B
  proc.jetPuppiTable.variables.btagRobustParTAK4CvL = ROBUSTPARTAK4VARS.btagRobustParTAK4CvL
  proc.jetPuppiTable.variables.btagRobustParTAK4CvB = ROBUSTPARTAK4VARS.btagRobustParTAK4CvB
  #
  # Save UnifiedParTAK4 b-tagging and c-tagging variables
  #
  proc.jetPuppiTable.variables.btagUParTAK4B = UNIFIEDPARTAK4VARS.btagUParTAK4B
  proc.jetPuppiTable.variables.btagUParTAK4CvL = UNIFIEDPARTAK4VARS.btagUParTAK4CvL
  proc.jetPuppiTable.variables.btagUParTAK4CvB = UNIFIEDPARTAK4VARS.btagUParTAK4CvB
  proc.jetPuppiTable.variables.btagUParTAK4SvCB = UNIFIEDPARTAK4VARS.btagUParTAK4SvCB
  proc.jetPuppiTable.variables.btagUParTAK4SvUDG = UNIFIEDPARTAK4VARS.btagUParTAK4SvUDG
  proc.jetPuppiTable.variables.btagUParTAK4UDG = UNIFIEDPARTAK4VARS.btagUParTAK4UDG
  proc.jetPuppiTable.variables.btagUParTAK4QvG = UNIFIEDPARTAK4VARS.btagUParTAK4QvG
  proc.jetPuppiTable.variables.btagUParTAK4TauVJet = UNIFIEDPARTAK4VARS.btagUParTAK4TauVJet
  proc.jetPuppiTable.variables.UParTAK4RegPtRawCorr = UNIFIEDPARTAK4VARS.UParTAK4RegPtRawCorr
  proc.jetPuppiTable.variables.UParTAK4RegPtRawCorrNeutrino = UNIFIEDPARTAK4VARS.UParTAK4RegPtRawCorrNeutrino
  proc.jetPuppiTable.variables.UParTAK4RegPtRawRes = UNIFIEDPARTAK4VARS.UParTAK4RegPtRawRes
  #
  # For Run-2 eras, don't need to save the low pt AK4 Puppi jet table for MET
  #
  run2_nanoAOD_ANY.toReplaceWith(
    proc.jetPuppiForMETTask,
    proc.jetPuppiForMETTask.copyAndExclude([proc.corrT1METJetPuppiTable])
  )

  #
  # Save MC-only jet variables in jet table
  #
  if runOnMC:
    jetMCTableName = "jet{}MCTable".format(jetName)
    setattr(proc, jetMCTableName, proc.jetMCTable.clone(
        src = proc.jetPuppiTable.src,
        name = proc.jetPuppiTable.name
      )
    )
    getattr(proc,jetMCTableName).variables.genJetIdx = PFJETVARS.genJetIdx

    jetMCTableTaskName = "jet{}MCTablesTask".format(jetName)
    setattr(proc, jetMCTableTaskName, cms.Task(getattr(proc,jetMCTableName)))

    run2_nanoAOD_ANY.toReplaceWith(
      proc.nanoTableTaskFS,
      proc.nanoTableTaskFS.copyAndAdd( getattr(proc,jetMCTableTaskName))
    )

  return proc

def ReclusterAK4CHSJets(proc, recoJA, runOnMC):
  """
  Recluster AK4 CHS jets and replace slimmedJets that is used as default to
  save AK4 CHS jets in NanoAODs (for Run-2).
  """
  print("custom_jme_cff::ReclusterAK4CHSJets: Recluster AK4 PF CHS jets")

  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll

  bTagDiscriminatorsForAK4CHS = cms.PSet(
    foo = cms.vstring(_pfParticleNetFromMiniAODAK4CHSCentralJetTagsAll+_pfParticleNetFromMiniAODAK4CHSForwardJetTagsAll)
  )
  bTagDiscriminatorsForAK4CHS = bTagDiscriminatorsForAK4CHS.foo.value()

  #
  # Recluster AK4 CHS jets
  #
  cfg = {
    "jet" : "ak4pfchs",
    "inputCollection" : "",
    "genJetsCollection": "AK4GenJetsNoNu",
    "bTagDiscriminators": bTagDiscriminatorsForAK4CHS,
    "minPtFastjet" : 0.,
  }
  recoJetInfo = recoJA.addRecoJetCollection(proc, **cfg)

  jetName = recoJetInfo.jetUpper
  patJetFinalColl = recoJetInfo.patJetFinalCollection

  #
  # Change the input jet source for jetCorrFactorsNano
  # and updatedJets
  #
  proc.jetCorrFactorsNano.src=patJetFinalColl
  proc.updatedJets.jetSource=patJetFinalColl

  #
  # Change pt cut
  #
  finalJetsCut = ""
  if runOnMC:
    finalJetsCut = "(pt >= 10) || ((pt < 10) && (genJetFwdRef().backRef().isNonnull()))"
  else:
    finalJetsCut = "(pt >= 10)"

  proc.finalJets.cut = finalJetsCut
  #
  # Add a minimum pt cut for corrT1METJets.
  #
  proc.corrT1METJetTable.cut = "pt>=10 && pt<15 && abs(eta)<9.9"

  #
  # Jet table cut
  #
  jetTableCut = "" # must not have any cut at the jetTable for AK4 CHS as it has been cross-cleaned
  proc.jetTable.src   = cms.InputTag("finalJets")
  proc.jetTable.cut   = jetTableCut
  proc.jetMCTable.cut = jetTableCut
  proc.jetTable.name  = "JetCHS"

  #
  # For Run-2 eras, the main AK4 jet collection in NanoAOD is the CHS collection
  #
  run2_nanoAOD_ANY.toModify(
    proc.jetTable,
    src = cms.InputTag("linkedObjects","jets"),
    name = "Jet"
  )

  #
  # Jet table documentation
  #
  jetTableDoc = "AK4 PF CHS jets with JECs applied. Jets with pt >= 10 GeV are stored."
  if runOnMC:
    jetTableDoc += "For jets with pt < 10 GeV, only those matched to AK4 Gen jets are stored."
  proc.jetTable.doc   = jetTableDoc

  proc.jetTable.variables.rawFactor.precision = 10

  #
  # Add variables
  #
  proc.jetTable.variables.chHadMultiplicity = PFJETVARS.chHadMultiplicity
  proc.jetTable.variables.neHadMultiplicity = PFJETVARS.neHadMultiplicity
  proc.jetTable.variables.hfHadMultiplicity = PFJETVARS.hfHadMultiplicity
  proc.jetTable.variables.hfEMMultiplicity  = PFJETVARS.hfEMMultiplicity
  proc.jetTable.variables.muMultiplicity    = PFJETVARS.muMultiplicity
  proc.jetTable.variables.elMultiplicity    = PFJETVARS.elMultiplicity
  proc.jetTable.variables.phoMultiplicity   = PFJETVARS.phoMultiplicity

  #
  # Add charged energy fraction from other primary vertices
  #
  proc.updatedJetsWithUserData.userFloats.chFPV1EF = cms.InputTag("jercVars:chargedFromPV1EnergyFraction")
  proc.updatedJetsWithUserData.userFloats.chFPV2EF = cms.InputTag("jercVars:chargedFromPV2EnergyFraction")
  proc.updatedJetsWithUserData.userFloats.chFPV3EF = cms.InputTag("jercVars:chargedFromPV3EnergyFraction")
  proc.jetTable.variables.chFPV1EF = Var("userFloat('chFPV1EF')", float, doc="charged fromPV==1 Energy Fraction (component of the total charged Energy Fraction).", precision= 6)
  proc.jetTable.variables.chFPV2EF = Var("userFloat('chFPV2EF')", float, doc="charged fromPV==2 Energy Fraction (component of the total charged Energy Fraction).", precision= 6)
  proc.jetTable.variables.chFPV3EF = Var("userFloat('chFPV3EF')", float, doc="charged fromPV==3 Energy Fraction (component of the total charged Energy Fraction).", precision= 6)

  #
  # Remove these tagger branches since for CHS, we just want to store
  # one tagger only.
  #
  for varNames in proc.jetTable.variables.parameterNames_():
    if "btagDeepFlav" in varNames or "btagRobustParT" in varNames or "btagUParT" in varNames:
      delattr(proc.jetTable.variables, varNames)
    if "UParTAK4Reg" in varNames:
      delattr(proc.jetTable.variables, varNames)
    if "svIdx" in varNames or "muonIdx" in varNames:
      delattr(proc.jetTable.variables, varNames)
    if "nElectrons" in varNames or "nMuons" in varNames or "nSVs" in varNames:
      delattr(proc.jetTable.variables, varNames)

  proc.jetTable.variables.btagPNetB = Var("?pt>15 && bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:BvsAll')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:BvsAll'):-1",float,precision=10,doc="ParticleNet b vs. udscg")
  proc.jetTable.variables.btagPNetCvL = Var("?pt>15 && bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsL')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsL'):-1",float,precision=10,doc="ParticleNet c vs. udsg")
  proc.jetTable.variables.btagPNetCvB = Var("?pt>15 && bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsB')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:CvsB'):-1",float,precision=10,doc="ParticleNet c vs. b")
  proc.jetTable.variables.btagPNetQvG = Var("?pt>15 && abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:QvsG'):bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardDiscriminatorsJetTags:QvsG')",float,precision=10,doc="ParticleNet q (udsbc) vs. g")
  proc.jetTable.variables.btagPNetTauVJet = Var("?pt>15 && bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:TauVsJet')>0?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralDiscriminatorsJetTags:TauVsJet'):-1",float,precision=10,doc="ParticleNet tau vs. jet")
  proc.jetTable.variables.PNetRegPtRawCorr = Var("?pt>15 && abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptcorr'):bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptcorr')",float,precision=10,doc="ParticleNet universal flavor-aware visible pT regression (no neutrinos), correction relative to raw jet pT")
  proc.jetTable.variables.PNetRegPtRawCorrNeutrino = Var("?pt>15 && abs(eta())<2.5?bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptnu'):bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptnu')",float,precision=10,doc="ParticleNet universal flavor-aware pT regression neutrino correction, relative to visible. To apply full regression, multiply raw jet pT by both PNetRegPtRawCorr and PNetRegPtRawCorrNeutrino.")
  proc.jetTable.variables.PNetRegPtRawRes = Var("?pt>15 && abs(eta())<2.5?0.5*(bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptreshigh')-bDiscriminator('pfParticleNetFromMiniAODAK4CHSCentralJetTags:ptreslow')):0.5*(bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptreshigh')-bDiscriminator('pfParticleNetFromMiniAODAK4CHSForwardJetTags:ptreslow'))",float,precision=10,doc="ParticleNet universal flavor-aware jet pT resolution estimator, (q84 - q16)/2")


  #Adding hf shower shape producer to the jet sequence. By default this producer is not automatically rerun at the NANOAOD step
  #The following lines make sure it is.
  hfJetShowerShapeforCustomNanoAOD = "hfJetShowerShapeforCustomNanoAOD"
  setattr(proc, hfJetShowerShapeforCustomNanoAOD, hfJetShowerShape.clone(jets="updatedJets", vertices="offlineSlimmedPrimaryVertices") )
  proc.jetUserDataTask.add(getattr(proc, hfJetShowerShapeforCustomNanoAOD))
  proc.updatedJetsWithUserData.userFloats.hfsigmaEtaEta = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:sigmaEtaEta')
  proc.updatedJetsWithUserData.userFloats.hfsigmaPhiPhi = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:sigmaPhiPhi')
  proc.updatedJetsWithUserData.userInts.hfcentralEtaStripSize = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:centralEtaStripSize')
  proc.updatedJetsWithUserData.userInts.hfadjacentEtaStripsSize = cms.InputTag('hfJetShowerShapeforCustomNanoAOD:adjacentEtaStripsSize')
  proc.jetTable.variables.hfsigmaEtaEta = Var("userFloat('hfsigmaEtaEta')",float,doc="sigmaEtaEta for HF jets (noise discriminating variable)",precision=10)
  proc.jetTable.variables.hfsigmaPhiPhi = Var("userFloat('hfsigmaPhiPhi')",float,doc="sigmaPhiPhi for HF jets (noise discriminating variable)",precision=10)
  proc.jetTable.variables.hfcentralEtaStripSize = Var("userInt('hfcentralEtaStripSize')", int, doc="eta size of the central tower strip in HF (noise discriminating variable) ")
  proc.jetTable.variables.hfadjacentEtaStripsSize = Var("userInt('hfadjacentEtaStripsSize')", int, doc="eta size of the strips next to the central tower strip in HF (noise discriminating variable) ")

  #
  # Since AK4 Puppi jet is the main AK4 jet collection for Run-3, disable
  # b-jets/c-jets NN-based mass regression for AK4 CHS.
  #
  (~run2_nanoAOD_ANY).toReplaceWith(
    proc.jetUserDataTask,
    proc.jetUserDataTask.copyAndExclude([proc.bJetVars])
  ).toReplaceWith(
    proc.jetTablesTask,
    proc.jetTablesTask.copyAndExclude([proc.bjetNN, proc.cjetNN])
  ).toModify(proc.updatedJetsWithUserData.userFloats,
    leadTrackPt = None,
    leptonPtRelv0 = None,
    leptonPtRelInvv0 = None,
    leptonDeltaR = None,
    vtxPt = None,
    vtxMass = None,
    vtx3dL = None,
    vtx3deL = None,
    ptD = None,
  ).toModify(
    proc.updatedJetsWithUserData.userInts,
    vtxNtrk = None,
    leptonPdgId = None
  ).toModify(
    proc.jetTable, externalVariables = cms.PSet()
  ).toReplaceWith(
  #
  # For Run-3, don't need to save the low pt AK4 CHS jet table for MET
  #
    proc.jetForMETTask,
    proc.jetForMETTask.copyAndExclude([proc.corrT1METJetTable])
  )

  #
  # Save MC-only jet variables in jet table
  #
  if runOnMC:
    jetMCTableName = "jet{}MCTable".format(jetName)
    setattr(proc, jetMCTableName, proc.jetMCTable.clone(
        src = proc.jetTable.src,
        name = proc.jetTable.name
      )
    )
    getattr(proc,jetMCTableName).variables.genJetIdx = PFJETVARS.genJetIdx

    jetMCTableTaskName = "jet{}MCTablesTask".format(jetName)
    setattr(proc, jetMCTableTaskName, cms.Task(getattr(proc,jetMCTableName)))

    (~run2_nanoAOD_ANY).toReplaceWith(
      proc.nanoTableTaskFS,
      proc.nanoTableTaskFS.copyAndAdd(getattr(proc,jetMCTableTaskName))
    )

  return proc

def AddNewAK8PuppiJetsForJEC(proc, recoJA, runOnMC):
  """
  Store a separate AK8 Puppi jet collection for JEC studies.
  Only minimal info are stored
  """
  print("custom_jme_cff::AddNewAK8PuppiJetsForJEC: Make a new AK8 PF Puppi jet collection for JEC studies")

  #
  # Recluster AK8 Puppi jets
  #
  cfg = {
    "jet" : "ak8pfpuppi",
    "inputCollection" : "",
    "genJetsCollection": "AK8GenJetsNoNu",
    "minPtFastjet" : 0., # Remove any pt threshold at the jet clustering stage.
  }
  recoJetInfo = recoJA.addRecoJetCollection(proc, **cfg)

  jetName = recoJetInfo.jetUpper
  payload = recoJetInfo.jetCorrPayload

  patJetFinalColl = recoJetInfo.patJetFinalCollection
  jetTablePrefix  = "FatJetForJEC"
  jetTableDoc     = "AK8 PF Puppi jets with JECs applied. Reclustered for JEC studies so only minimal info stored."
  ptcut           = 15

  SavePatJets(proc,
    jetName, payload, patJetFinalColl, jetTablePrefix, jetTableDoc, doPF=True,
    doCalo=False, ptcut=ptcut, doPUIDVar=False, doQGL=False, doBTag=False, runOnMC=runOnMC
  )

  return proc

def AddNewAK8CHSJets(proc, recoJA, runOnMC):
  """
  Store an AK8 CHS jet collection for JEC studies.
  """
  print("custom_jme_cff::AddNewAK8CHSJets: Make a new AK8 PF CHS jet collection for JEC studies")

  #
  # Recluster AK8 CHS jets
  #
  cfg = {
    "jet" : "ak8pfchs",
    "inputCollection" : "",
    "genJetsCollection": "AK8GenJetsNoNu",
    "minPtFastjet" : 0., # Remove any pt threshold at the jet clustering stage.
  }
  recoJetInfo = recoJA.addRecoJetCollection(proc, **cfg)

  jetName = recoJetInfo.jetUpper
  payload = recoJetInfo.jetCorrPayload

  patJetFinalColl = recoJetInfo.patJetFinalCollection
  jetTablePrefix  = "FatJetCHS"
  jetTableDoc     = "AK8 PF CHS jets with JECs applied. Reclustered for JEC studies so only minimal info stored."
  ptcut           = 15

  SavePatJets(proc,
    jetName, payload, patJetFinalColl, jetTablePrefix, jetTableDoc, doPF=True,
    doCalo=False, ptcut=ptcut, doPUIDVar=False, doQGL=False, doBTag=False, runOnMC=runOnMC
  )

  return proc

def AddVariablesForAK8PuppiJets(proc):
  """
  Add more variables for AK8 PFPUPPI jets
  """

  proc.fatJetTable.variables.rawFactor.precision = 10

  #
  #  These variables are not stored for AK8PFPUPPI (slimmedJetsAK8)
  #  in MiniAOD if their pt < 170 GeV. Hence the conditional fill.
  #
  proc.fatJetTable.variables.chHEF  = Var("?isPFJet()?chargedHadronEnergyFraction():-1", float, doc="charged Hadron Energy Fraction", precision = 6)
  proc.fatJetTable.variables.neHEF  = Var("?isPFJet()?neutralHadronEnergyFraction():-1", float, doc="neutral Hadron Energy Fraction", precision = 6)
  proc.fatJetTable.variables.chEmEF = Var("?isPFJet()?chargedEmEnergyFraction():-1", float, doc="charged Electromagnetic Energy Fraction", precision = 6)
  proc.fatJetTable.variables.neEmEF = Var("?isPFJet()?neutralEmEnergyFraction():-1", float, doc="neutral Electromagnetic Energy Fraction", precision = 6)
  proc.fatJetTable.variables.muEF   = Var("?isPFJet()?muonEnergyFraction():-1", float, doc="muon Energy Fraction", precision = 6)
  proc.fatJetTable.variables.hfHEF  = Var("?isPFJet()?HFHadronEnergyFraction():-1", float, doc="energy fraction in forward hadronic calorimeter", precision = 6)
  proc.fatJetTable.variables.hfEmEF = Var("?isPFJet()?HFEMEnergyFraction():-1", float, doc="energy fraction in forward EM calorimeter", precision = 6)
  proc.fatJetTable.variables.chHadMultiplicity = Var("?isPFJet()?chargedHadronMultiplicity():-1","int16", doc="(Puppi-weighted) number of charged hadrons in the jet")
  proc.fatJetTable.variables.neHadMultiplicity = Var("?isPFJet()?neutralHadronMultiplicity():-1","int16", doc="(Puppi-weighted) number of neutral hadrons in the jet")
  proc.fatJetTable.variables.hfHadMultiplicity = Var("?isPFJet()?HFHadronMultiplicity():-1", "int16", doc="(Puppi-weighted) number of HF Hadrons in the jet")
  proc.fatJetTable.variables.hfEMMultiplicity  = Var("?isPFJet()?HFEMMultiplicity():-1", "int16", doc="(Puppi-weighted) number of HF EMs in the jet")
  proc.fatJetTable.variables.muMultiplicity    = Var("?isPFJet()?muonMultiplicity():-1", "int16", doc="(Puppi-weighted) number of muons in the jet")
  proc.fatJetTable.variables.elMultiplicity    = Var("?isPFJet()?electronMultiplicity():-1", "int16", doc="(Puppi-weighted) number of electrons in the jet")
  proc.fatJetTable.variables.phoMultiplicity   = Var("?isPFJet()?photonMultiplicity():-1", "int16", doc="(Puppi-weighted) number of photons in the jet")

  return proc
#******************************************
#
#
# Gen Jets related functions
#
#
#******************************************
def AddNewGenJets(proc, genJetInfo):
  """
  Add genJet into custom nanoAOD
  """

  genJetName         = genJetInfo.jetUpper
  genJetAlgo         = genJetInfo.jetAlgo
  genJetSize         = genJetInfo.jetSize
  genJetSizeNr       = genJetInfo.jetSizeNr
  genJetFinalColl    = "{}{}{}".format(genJetAlgo.upper(), genJetSize, "GenJetsNoNu")
  genJetTablePrefix  = nanoInfo_genjets[genJetInfo.jet]["name"]
  genJetTableDoc     = nanoInfo_genjets[genJetInfo.jet]["doc"]

  SaveGenJets(proc, genJetName, genJetAlgo, genJetSizeNr, genJetFinalColl, genJetTablePrefix, genJetTableDoc, runOnMC=False)

  return proc

def SaveGenJets(proc, genJetName, genJetAlgo, genJetSizeNr, genJetFinalColl, genJetTablePrefix, genJetTableDoc, runOnMC=False):
  """
  Schedule modules for a given genJet collection and save its variables into custom NanoAOD
  """

  genJetTableName = "jet{}Table".format(genJetName)
  setattr(proc, genJetTableName, genJetTable.clone(
      src       = genJetFinalColl,
      cut       = "", # No cut specified here. Save all gen jets after clustering
      name      = genJetTablePrefix,
      doc       = genJetTableDoc,
      variables = GENJETVARS
    )
  )

  genJetFlavourAssociationName = "genJet{}FlavourAssociation".format(genJetName)
  setattr(proc, genJetFlavourAssociationName, genJetFlavourAssociation.clone(
      jets           = getattr(proc,genJetTableName).src,
      jetAlgorithm   = supportedJetAlgos[genJetAlgo],
      rParam         = genJetSizeNr,
    )
  )

  genJetFlavourTableName = "genJet{}FlavourTable".format(genJetName)
  setattr(proc, genJetFlavourTableName, genJetFlavourTable.clone(
      name            = getattr(proc,genJetTableName).name,
      src             = getattr(proc,genJetTableName).src,
      cut             = getattr(proc,genJetTableName).cut,
      jetFlavourInfos = genJetFlavourAssociationName,
    )
  )

  genJetTaskName = "genJet{}Task".format(genJetName)
  setattr(proc, genJetTaskName, cms.Task(
      getattr(proc,genJetTableName),
      getattr(proc,genJetFlavourAssociationName),
      getattr(proc,genJetFlavourTableName)
    )
  )
  proc.jetMCTask.add(getattr(proc,genJetTaskName))

  return proc

def ReclusterAK4GenJets(proc, genJA):
  """
  Recluster AK4 Gen jets and replace
  slimmedGenJets that is used as default
  to save AK4 Gen jets in NanoAODs.
  """
  print("custom_jme_cff::ReclusterAK4GenJets: Recluster AK4 Gen jets")

  #
  # Recluster AK4 Gen jet
  #
  cfg = {
    "jet" : "ak4gen",
    "minPtFastjet" : 5.
  }
  genJetInfo = genJA.addGenJetCollection(proc, **cfg)

  genJetName      = genJetInfo.jetUpper
  genJetAlgo      = genJetInfo.jetAlgo
  genJetSize      = genJetInfo.jetSize
  genJetSizeNr    = genJetInfo.jetSizeNr
  selectedGenJets = "{}{}{}".format(genJetAlgo.upper(), genJetSize, "GenJetsNoNu")

  #
  # Change jet source to the newly clustered jet collection. Set very low pt cut for jets
  # to be stored in the GenJet Table
  #
  proc.genJetTable.src = selectedGenJets
  proc.genJetTable.cut = "" # No cut specified here. Save all gen jets after clustering
  proc.genJetTable.doc = "AK4 Gen jets (made with visible genparticles) with pt > 5 GeV"

  genJetFlavourAssociationName = "genJet{}FlavourAssociation".format(genJetName)
  setattr(proc, genJetFlavourAssociationName, genJetFlavourAssociation.clone(
      jets           = proc.genJetTable.src,
      jetAlgorithm   = supportedJetAlgos[genJetAlgo],
      rParam         = genJetSizeNr,
    )
  )
  proc.jetMCTask.add(getattr(proc, genJetFlavourAssociationName))
  return proc

def AddNewAK8GenJetsForJEC(proc, genJA):
  """
  Make a separate AK8 Gen jet collection for JEC studies.
  """
  print("custom_jme_cff::AddNewAK8GenJetsForJEC: Add new AK8 Gen jets for JEC studies")

  #
  # Recluster AK8 Gen jet
  #
  cfg = {
    "jet" : "ak8gen",
    "minPtFastjet" : 10.
  }
  genJetInfo = genJA.addGenJetCollection(proc, **cfg)

  genJetName         = genJetInfo.jetUpper
  genJetAlgo         = genJetInfo.jetAlgo
  genJetSize         = genJetInfo.jetSize
  genJetSizeNr       = genJetInfo.jetSizeNr
  genJetFinalColl    = "{}{}{}".format(genJetAlgo.upper(), genJetSize, "GenJetsNoNu")
  genJetTablePrefix  = "GenJetAK8ForJEC"
  genJetTableDoc     = "AK8 Gen jets (made with visible genparticles) with pt > 10 GeV. Reclustered for JEC studies."

  SaveGenJets(proc, genJetName, genJetAlgo, genJetSizeNr, genJetFinalColl, genJetTablePrefix, genJetTableDoc, runOnMC=False)

  return proc

def AddVariablesForAK4GenJets(proc):
  proc.genJetTable.variables.nConstituents = GENJETVARS.nConstituents
  return proc

def AddVariablesForAK8GenJets(proc):
  proc.genJetAK8Table.variables.nConstituents = GENJETVARS.nConstituents
  return proc

def ModifyAK4JetMCTable(proc):
  # Remove the genjet pt selection when saving the genJetIdx, which is
  # the default in main Nano
  proc.jetMCTable.variables.genJetIdx = PFJETVARS.genJetIdx
  return proc

#===========================================================================
#
# Misc. functions
#
#===========================================================================
def RemoveAllJetPtCuts(proc):
  """
  Remove default pt cuts for all jets set in jets_cff.py
  """

  proc.finalJets.cut             = "" # 15 -> 10
  proc.finalJetsPuppi.cut        = "" # 15 -> 10
  proc.finalJetsAK8.cut          = "" # 170 -> 170
  proc.genJetTable.cut           = "" # 10 -> 8
  proc.genJetFlavourTable.cut    = "" # 10 -> 8
  proc.genJetAK8Table.cut        = "" # 100 -> 80
  proc.genJetAK8FlavourTable.cut = "" # 100 -> 80

  return proc

def RecomputePuppiWeights(proc):
  """
  Setup packedpuppi and packedpuppiNoLep to recompute puppi weights
  """
  if hasattr(proc,"packedpuppi"):
    proc.packedpuppi.useExistingWeights = False
  if hasattr(proc,"packedpuppiNoLep"):
    proc.packedpuppiNoLep.useExistingWeights = False
  return proc

def RecomputePuppiMET(proc):
  """
  Recompute PuppiMET. This is useful when puppi weights are recomputed.
  """
  runOnMC=True
  if hasattr(proc,"NANOEDMAODoutput") or hasattr(proc,"NANOAODoutput"):
    runOnMC = False

  from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
  runMetCorAndUncFromMiniAOD(proc, isData=not(runOnMC),
    jetCollUnskimmed='updatedJetsPuppi',metType='Puppi',postfix='Puppi',jetFlavor='AK4PFPuppi',
    puppiProducerLabel='packedpuppi',puppiProducerForMETLabel='packedpuppiNoLep',
    recoMetFromPFCs=True
  )
  return proc

#===========================================================================
#
# Functions to setup recomputation of Puppi weights and recluster PuppiMET
# and AK8 Puppi jets (slimmedJetsAK8)
#
#===========================================================================
def RecomputePuppiWeightsAndMET(proc):
  """
  Recompute Puppi weights and PuppiMET.
  """

  proc = RecomputePuppiWeights(proc)
  proc = RecomputePuppiMET(proc)

  return proc

def RecomputePuppiWeightsMETAK8(proc):
  """
  Recompute Puppi weights and PuppiMET and rebuild slimmedJetsAK8.
  """
  runOnMC=True
  if hasattr(proc,"NANOEDMAODoutput") or hasattr(proc,"NANOAODoutput"):
    runOnMC = False

  proc = RecomputePuppiWeightsAndMET(proc)

  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetJetTagsAll as pfParticleNetJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassRegressionOutputs as pfParticleNetMassRegressionOutputs
  from RecoBTag.ONNXRuntime.pfParticleNet_cff import _pfParticleNetMassCorrelatedJetTagsAll as pfParticleNetMassCorrelatedJetTagsAll
  from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK8_cff import _pfParticleNetFromMiniAODAK8JetTagsAll as pfParticleNetFromMiniAODAK8JetTagsAll

  btagDiscriminatorsAK8 = cms.PSet(names = cms.vstring(
      pfParticleNetMassCorrelatedJetTagsAll+
      pfParticleNetFromMiniAODAK8JetTagsAll+
      pfParticleNetJetTagsAll+
      pfParticleNetMassRegressionOutputs
    )
  )

  btagDiscriminatorsAK8Subjets = cms.PSet(names = cms.vstring(
      'pfDeepFlavourJetTags:probb',
      'pfDeepFlavourJetTags:probbb',
      'pfDeepFlavourJetTags:problepb',
      'pfUnifiedParticleTransformerAK4DiscriminatorsJetTags:BvsAll'
    )
  )

  run3_nanoAOD_pre142X.toModify(btagDiscriminatorsAK8Subjets,
    names = cms.vstring('pfDeepCSVJetTags:probb','pfDeepCSVJetTags:probbb')
  )

  from PhysicsTools.PatAlgos.tools.puppiJetMETReclusteringFromMiniAOD_cff import setupPuppiAK4AK8METReclustering
  proc = setupPuppiAK4AK8METReclustering(proc, runOnMC=runOnMC, useExistingWeights=False,
    reclusterAK4MET=False, # Already setup to recluster AK4 Puppi jets and PuppiMET
    reclusterAK8=True,
    btagDiscriminatorsAK8=btagDiscriminatorsAK8,
    btagDiscriminatorsAK8Subjets=btagDiscriminatorsAK8Subjets
  )

  return proc

#===========================================================================
#
# CUSTOMIZATION function
#
#===========================================================================
def PrepJMECustomNanoAOD(process):
  ## TODO : find a better way to handle data or MC by modifying the proper Tasks
  runOnMC=True
  if hasattr(process,"NANOEDMAODoutput") or hasattr(process,"NANOAODoutput"):
    runOnMC = False
  ############################################################################
  # Remove all default jet pt cuts from jets_cff.py
  ############################################################################
  process = RemoveAllJetPtCuts(process)

  ###########################################################################
  #
  # Gen-level jets related functions. Only for MC.
  #
  ###########################################################################
  genJA = GenJetAdder()
  if runOnMC:
    ############################################################################
    # Save additional variables for AK8 GEN jets
    ############################################################################
    process = AddVariablesForAK8GenJets(process)
    ############################################################################
    # Recluster AK8 GEN jets
    ############################################################################
    process = AddNewAK8GenJetsForJEC(process, genJA)
    ############################################################################
    # Modify jetMCTable
    ############################################################################
    process = ModifyAK4JetMCTable(process)
    ###########################################################################
    # Recluster AK4 GEN jets
    ###########################################################################
    process = ReclusterAK4GenJets(process, genJA)
    process = AddVariablesForAK4GenJets(process)
    ###########################################################################
    # Add additional GEN jets to NanoAOD
    ###########################################################################
    for jetConfig in config_genjets:
      cfg = { k : v for k, v in jetConfig.items() if k != "enabled"}
      genJetInfo = genJA.addGenJetCollection(process, **cfg)
      AddNewGenJets(process, genJetInfo)

  ###########################################################################
  #
  # Reco-level jets related functions. For both MC and data.
  #
  ###########################################################################
  recoJA = RecoJetAdder(runOnMC=runOnMC)
  ###########################################################################
  # Save additional variables for AK8Puppi jets
  ###########################################################################
  process = AddVariablesForAK8PuppiJets(process)
  ###########################################################################
  # Build a separate AK8Puppi jet collection for JEC studies
  ###########################################################################
  process = AddNewAK8PuppiJetsForJEC(process, recoJA, runOnMC)
  ###########################################################################
  # Recluster AK4 CHS jets and replace "slimmedJets"
  ###########################################################################
  process = ReclusterAK4CHSJets(process, recoJA, runOnMC)
  ###########################################################################
  # Recluster AK4 Puppi jets and replace "slimmedJets"
  ###########################################################################
  process = ReclusterAK4PuppiJets(process, recoJA, runOnMC)
  ###########################################################################
  # Add additional Reco jets to NanoAOD
  ###########################################################################
  for jetConfig in config_recojets:
    cfg = { k : v for k, v in jetConfig.items() if k != "enabled"}
    recoJetInfo = recoJA.addRecoJetCollection(process, **cfg)
    AddNewPatJets(process, recoJetInfo, runOnMC)

  ###########################################################################
  # Add jet tasks
  # By default for Run-3, add AK4 CHS jet tasks.
  # For Run-2 eras, add AK4 Puppi jet tasks
  ###########################################################################
  def addAK4JetTasks(proc, addAK4CHSJetTasks, addAK4PuppiJetTasks):
    if addAK4CHSJetTasks:
      proc.nanoTableTaskCommon.add(proc.jetTask)
      proc.nanoTableTaskCommon.add(proc.jetTablesTask)
      proc.nanoTableTaskCommon.add(proc.jetForMETTask)
    if addAK4PuppiJetTasks:
      proc.nanoTableTaskCommon.add(proc.jetPuppiTask)
      proc.nanoTableTaskCommon.add(proc.jetPuppiTablesTask)
      proc.nanoTableTaskCommon.add(proc.jetPuppiForMETTask)
    return proc

  jmeNano_addAK4JetTasks_switch = cms.PSet(
    jmeNano_addAK4CHS_switch = cms.untracked.bool(True),
    jmeNano_addAK4Puppi_switch = cms.untracked.bool(False)
  )
  run2_nanoAOD_ANY.toModify(jmeNano_addAK4JetTasks_switch,
    jmeNano_addAK4CHS_switch = False,
    jmeNano_addAK4Puppi_switch = True
  )
  process = addAK4JetTasks(process,
    addAK4CHSJetTasks = jmeNano_addAK4JetTasks_switch.jmeNano_addAK4CHS_switch,
    addAK4PuppiJetTasks = jmeNano_addAK4JetTasks_switch.jmeNano_addAK4Puppi_switch,
  )

  ###########################################################################
  # Save Maximum of Pt Hat Max
  ###########################################################################
  if runOnMC:
    process.puTable.savePtHatMax = True

  ###########################################################################
  # Save all Parton-Shower weights
  ###########################################################################
  if runOnMC:
    process.genWeightsTable.keepAllPSWeights = True

  return process

