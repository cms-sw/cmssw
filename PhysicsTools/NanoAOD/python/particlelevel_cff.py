import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer
from PhysicsTools.NanoAOD.simpleSingletonCandidateFlatTableProducer_cfi import simpleSingletonCandidateFlatTableProducer
from PhysicsTools.NanoAOD.simpleHTXSFlatTableProducer_cfi import simpleHTXSFlatTableProducer

##################### User floats producers, selectors ##########################

mergedGenParticles = cms.EDProducer("MergedGenParticleProducer",
    inputPruned = cms.InputTag("prunedGenParticles"),
    inputPacked = cms.InputTag("packedGenParticles"),
)

genParticles2HepMC = cms.EDProducer("GenParticles2HepMCConverter",
    genParticles = cms.InputTag("mergedGenParticles"),
    genEventInfo = cms.InputTag("generator"),
    signalParticlePdgIds = cms.vint32(),
)

genParticles2HepMCHiggsVtx = cms.EDProducer("GenParticles2HepMCConverter",
     genParticles = cms.InputTag("mergedGenParticles"),
     genEventInfo = cms.InputTag("generator"),
     signalParticlePdgIds = cms.vint32(25), ## for the Higgs analysis
)


particleLevel = cms.EDProducer("ParticleLevelProducer",
    src = cms.InputTag("genParticles2HepMC:unsmeared"),

    doJetClustering = cms.bool(False), # Not needed as Rivet jets aren't used currently
    usePromptFinalStates = cms.bool(True), # for leptons, photons, neutrinos
    excludePromptLeptonsFromJetClustering = cms.bool(False),
    excludeNeutrinosFromJetClustering = cms.bool(True),

    particleMinPt  = cms.double(0.),
    particleMaxEta = cms.double(5.), # HF range. Maximum 6.0 on MiniAOD

    lepConeSize = cms.double(0.1), # for photon dressing
    lepMinPt    = cms.double(1.),
    lepMaxEta   = cms.double(2.5),

    jetConeSize = cms.double(0.4),
    jetMinPt    = cms.double(10.),
    jetMaxEta   = cms.double(999.),

    fatJetConeSize = cms.double(0.8),
    fatJetMinPt    = cms.double(170.),
    fatJetMaxEta   = cms.double(999.),

    phoIsoConeSize = cms.double(0.4),
    phoMaxRelIso = cms.double(0.5),
    phoMinPt = cms.double(1.),
    phoMaxEta = cms.double(2.5),
)

rivetProducerHTXS = cms.EDProducer('HTXSRivetProducer',
   HepMCCollection = cms.InputTag('genParticles2HepMCHiggsVtx','unsmeared'),
   LHERunInfo = cms.InputTag('externalLHEProducer'),
   ProductionMode = cms.string('AUTO'),
)


##################### Tables for final output and docs ##########################
rivetLeptonTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("particleLevel:leptons"),
    cut = cms.string("pt > 15"),
    name= cms.string("GenDressedLepton"),
    doc = cms.string("Dressed leptons from Rivet-based ParticleLevelProducer"),
    externalVariables = cms.PSet(
        hasTauAnc = ExtVar(cms.InputTag("tautagger"),bool, doc="true if Dressed lepton has a tau as ancestor"),
        ),
    variables = cms.PSet(
        P4Vars,
        pdgId = Var("pdgId", int, doc="PDG id"),
    )
)

rivetPhotonTable = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("particleLevel:photons"),
    cut = cms.string("pt > 10"),
    name= cms.string("GenIsolatedPhoton"),
    doc = cms.string("Isolated photons from Rivet-based ParticleLevelProducer"),
    variables = cms.PSet(
        P4Vars
    )
)

tautagger = cms.EDProducer("GenJetTauTaggerProducer",
    src = rivetLeptonTable.src,
)

rivetMetTable = simpleSingletonCandidateFlatTableProducer.clone(
    src = cms.InputTag("particleLevel:mets"),
    name = cms.string("MET"),
    doc = cms.string("MET from Rivet-based ParticleLevelProducer in fiducial volume abs(eta)<5"),
    extension = cms.bool(True),
    variables = cms.PSet(
       fiducialGenPt  = Var("pt",  float, precision=10),
       fiducialGenPhi = Var("phi", float, precision=10),
    ),
)

HTXSCategoryTable = simpleHTXSFlatTableProducer.clone(
    src = cms.InputTag("rivetProducerHTXS","HiggsClassification"),
    name = cms.string("HTXS"),
    doc = cms.string("HTXS classification"),
    variables=cms.PSet(
        stage_0 = Var("stage0_cat",int, doc="HTXS stage-0 category"),
        stage_1_pTjet30 = Var("stage1_cat_pTjet30GeV",int, doc="HTXS stage-1 category (jet pt>30 GeV)"),
        stage_1_pTjet25 = Var("stage1_cat_pTjet25GeV",int, doc="HTXS stage-1 category (jet pt>25 GeV)"),
        stage1_1_cat_pTjet30GeV = Var("stage1_1_cat_pTjet30GeV",int,doc="HTXS stage-1.1 category(jet pt>30 GeV)"),
        stage1_1_cat_pTjet25GeV = Var("stage1_1_cat_pTjet25GeV",int,doc="HTXS stage-1.1 category(jet pt>25 GeV)"),
        stage1_1_fine_cat_pTjet30GeV = Var("stage1_1_fine_cat_pTjet30GeV",int,doc="HTXS stage-1.1-fine category(jet pt>30 GeV)"),
        stage1_1_fine_cat_pTjet25GeV = Var("stage1_1_fine_cat_pTjet25GeV",int,doc="HTXS stage-1.1-fine category(jet pt>25 GeV)"),
        stage1_2_cat_pTjet30GeV = Var("stage1_2_cat_pTjet30GeV",int,doc="HTXS stage-1.2 category(jet pt>30 GeV)"),
        stage1_2_cat_pTjet25GeV = Var("stage1_2_cat_pTjet25GeV",int,doc="HTXS stage-1.2 category(jet pt>25 GeV)"),
        stage1_2_fine_cat_pTjet30GeV = Var("stage1_2_fine_cat_pTjet30GeV",int,doc="HTXS stage-1.2-fine category(jet pt>30 GeV)"),
        stage1_2_fine_cat_pTjet25GeV = Var("stage1_2_fine_cat_pTjet25GeV",int,doc="HTXS stage-1.2-fine category(jet pt>25 GeV)"),
        Higgs_pt = Var("higgs.Pt()",float, doc="pt of the Higgs boson as identified in HTXS", precision=14),
        Higgs_y = Var("higgs.Rapidity()",float, doc="rapidity of the Higgs boson as identified in HTXS", precision=12),
        njets30 = Var("jets30.size()","uint8", doc="number of jets with pt>30 GeV as identified in HTXS"),
        njets25 = Var("jets25.size()","uint8", doc="number of jets with pt>25 GeV as identified in HTXS"),
   )
)

lheInfoTable = cms.EDProducer("LHETablesProducer",
     lheInfo = cms.VInputTag(cms.InputTag("externalLHEProducer"), cms.InputTag("source")),
     precision = cms.int32(14),
     storeLHEParticles = cms.bool(True)
 )

particleLevelTask = cms.Task(mergedGenParticles,genParticles2HepMC,particleLevel,tautagger,genParticles2HepMCHiggsVtx,rivetProducerHTXS)
particleLevelTablesTask = cms.Task(rivetLeptonTable,rivetPhotonTable,rivetMetTable,HTXSCategoryTable,lheInfoTable)
