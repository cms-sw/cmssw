import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltPFCandidateTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("hltParticleFlowTmp"),
    name = cms.string("hltPFCandidate"),
    cut = cms.string(""),
    doc = cms.string("HLT PF information"),
    singleton = cms.bool(False),
    extension = cms.bool(False), # this is the main table
    # externalVariables = cms.PSet(
    #   vertexIndex = ExtVar(cms.InputTag("hltParticleFlowTmp", "vertexIndex"), int, doc="vertex index"),
    #   trkNormchi2 = ExtVar(cms.InputTag("hltParticleFlowTmp", "normchi2"), float, doc="normalized chi squared of best track", precision=6),
    #   trkDz = ExtVar(cms.InputTag("hltParticleFlowTmp", "dz"), float, doc="dz of best track", precision=6),
    #   #  trkDxy = ExtVar(cms.InputTag("hltParticleFlowTmp", "dxy"), float, doc="dxy of best track", precision=6),
    #   trkDzsig = ExtVar(cms.InputTag("hltParticleFlowTmp", "dzsig"), float, doc="dzsig of best track", precision=6),
    #   trkDxysig = ExtVar(cms.InputTag("hltParticleFlowTmp", "dxysig"), float, doc="dxysig of best track", precision=6),
    #   trkLostInnerHits = ExtVar(cms.InputTag("hltParticleFlowTmp", "lostInnerHits"), int, doc="lostInnerHits of best track"),
    #   trkQuality = ExtVar(cms.InputTag("hltParticleFlowTmp", "quality"), int, doc="quality of best track"),
    #   trkPt = ExtVar(cms.InputTag("hltParticleFlowTmp", "trkPt"), float, doc="pt of best track", precision=6),
    #   trkEta = ExtVar(cms.InputTag("hltParticleFlowTmp", "trkEta"), float, doc="eta of best track", precision=6),
    #   trkPhi = ExtVar(cms.InputTag("hltParticleFlowTmp", "trkPhi"), float, doc="phi of best track", precision=6),
    # ),
    variables = cms.PSet(
      CandVars,
    ),
  )

hltJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
      src = cms.InputTag("hltAK4PFPuppiJets"),
      name = cms.string("hltAK4PuppiJet"),
      cut = cms.string(""),
      doc = cms.string("HLT PUPPI jets information"),
      singleton = cms.bool(False),
      extension = cms.bool(False), # this is the main table
      externalVariables = cms.PSet(
        DeepFlavour_prob_b = ExtVar(cms.InputTag("hltPfDeepFlavourJetTags:probb"), float, doc="DeepFlavour probability of b", precision=10),
        DeepFalvour_prob_bb = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probbb'), float, doc="ParticleNet probability of bb", precision=10),
        DeepFalvour_prob_c = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probc'), float, doc="ParticleNet probability of c", precision=10),
        DeepFalvour_prob_uds = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probuds'), float, doc="particlenet probability of uds", precision=10),
        DeepFalvour_prob_g = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:probg'), float, doc="ParticleNet probability of g", precision=10),
        DeepFalvour_prob_lepb = ExtVar(cms.InputTag('hltPfDeepFlavourJetTags:problepb'), float, doc="ParticleNet probability of lepb", precision=10),
      ),
      variables = cms.PSet(
        P4Vars,
        # area = Var("jetArea()", float, doc="jet catchment area, for JECs",precision=10),
        # chHEF = Var("chargedHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Hadron Energy Fraction", precision= 6),
        # neHEF = Var("neutralHadronEnergy()/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Hadron Energy Fraction", precision= 6),
        # chEmEF = Var("(electronEnergy()+muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="charged Electromagnetic Energy Fraction", precision= 6),
        # neEmEF = Var("(photonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="neutral Electromagnetic Energy Fraction", precision= 6),
        # muEF = Var("(muonEnergy())/(chargedHadronEnergy()+neutralHadronEnergy()+photonEnergy()+electronEnergy()+muonEnergy())", float, doc="muon Energy Fraction", precision= 6),
        # nCh = Var("chargedHadronMultiplicity()", int, doc="number of charged hadrons in the jet"),
        # nNh = Var("neutralHadronMultiplicity()", int, doc="number of neutral hadrons in the jet"),
        # nMuons = Var("muonMultiplicity()", int, doc="number of muons in the jet"),
        # nElectrons = Var("electronMultiplicity()", int, doc="number of electrons in the jet"),
        # nPhotons = Var("photonMultiplicity()", int, doc="number of photons in the jet"),
        nConstituents = Var("numberOfDaughters()", "uint8", doc="number of particles in the jet")
      ),
)