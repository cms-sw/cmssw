import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltPFCandidateTable = cms.EDProducer("SimplePFCandidateFlatTableProducer",
    skipNonExistingSrc = cms.bool(True),
    src = cms.InputTag("hltParticleFlowTmp"),
    name = cms.string("hltPFCandidate"),
    cut = cms.string(""),
    doc = cms.string("HLT PF information"),
    singleton = cms.bool(False),
    extension = cms.bool(False),
    variables = cms.PSet(
      CandVars,
      trackIndex = Var("trackRef().key()", "uint8", doc="track index")
    ),
  )

hltJetTable = cms.EDProducer("SimplePFJetFlatTableProducer",
      skipNonExistingSrc = cms.bool(True),
      src = cms.InputTag("hltAK4PFPuppiJets"),
      name = cms.string("hltAK4PuppiJet"),
      cut = cms.string(""),
      doc = cms.string("HLT PUPPI jets information"),
      singleton = cms.bool(False),
      extension = cms.bool(False),
      externalVariables = cms.PSet(
        DeepFlavour_prob_b = ExtVar(cms.InputTag("hltPfDeepFlavourJetTags:probb"), float, doc="DeepFlavour probability of b", precision=10),
        DeepFlavour_prob_bb = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probbb'), float, doc="DeepFlavour probability of bb", precision=10),
        DeepFlavour_prob_c = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probc'), float, doc="DeepFlavour probability of c", precision=10),
        DeepFlavour_prob_uds = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probuds'), float, doc="DeepFlavour probability of uds", precision=10),
        DeepFlavour_prob_g = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probg'), float, doc="DeepFlavour probability of g", precision=10),
        DeepFlavour_prob_lepb = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:problepb'), float, doc="DeepFlavour probability of lepb", precision=10),
      ),
      variables = cms.PSet(
        P4Vars,
        area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        chHEF = Var("chargedHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Hadron Energy Fraction", precision= 6),
        neHEF = Var("neutralHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Hadron Energy Fraction", precision= 6),
        chEmEF = Var("(electronEnergy()+muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        neEmEF = Var("(photonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        muEF = Var("(muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="muon Energy Fraction", precision= 6),
        nCh = Var("chargedHadronMultiplicity()", int, doc="number of charged hadrons in the jet"),
        nNh = Var("neutralHadronMultiplicity()", int, doc="number of neutral hadrons in the jet"),
        nMuons = Var("muonMultiplicity()", int, doc="number of muons in the jet"),
        nElectrons = Var("electronMultiplicity()", int, doc="number of electrons in the jet"),
        nPhotons = Var("photonMultiplicity()", int, doc="number of photons in the jet"),
        nConstituents = Var("numberOfDaughters()", "uint8", doc="number of particles in the jet")
      ),
)
