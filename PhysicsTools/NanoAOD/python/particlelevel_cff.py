import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.common_cff import *



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
    
    usePromptFinalStates = cms.bool(True), # for leptons, photons, neutrinos
    excludePromptLeptonsFromJetClustering = cms.bool(False),
    excludeNeutrinosFromJetClustering = cms.bool(True),
    
    particleMinPt  = cms.double(0.),
    particleMaxEta = cms.double(5.), # HF range. Maximum 6.0 on MiniAOD
    
    lepConeSize = cms.double(0.1), # for photon dressing
    lepMinPt    = cms.double(15.),
    lepMaxEta   = cms.double(2.5),
    
    jetConeSize = cms.double(0.4),
    jetMinPt    = cms.double(10.),
    jetMaxEta   = cms.double(999.),
    
    fatJetConeSize = cms.double(0.8),
    fatJetMinPt    = cms.double(170.),
    fatJetMaxEta   = cms.double(999.),

    phoIsoConeSize = cms.double(0.4),
    phoMaxRelIso = cms.double(0.5),
    phoMinPt = cms.double(10),
    phoMaxEta = cms.double(2.5),
)

rivetProducerHTXS = cms.EDProducer('HTXSRivetProducer',
   HepMCCollection = cms.InputTag('genParticles2HepMCHiggsVtx','unsmeared'),
   LHERunInfo = cms.InputTag('externalLHEProducer'),
   ProductionMode = cms.string('AUTO'),
)


##################### Tables for final output and docs ##########################
rivetLeptonTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("particleLevel:leptons"),
    cut = cms.string(""),
    name= cms.string("GenDressedLepton"),
    doc = cms.string("Dressed leptons from Rivet-based ParticleLevelProducer"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table
    externalVariables = cms.PSet(
        hasTauAnc = ExtVar(cms.InputTag("tautagger"),bool, doc="true if Dressed lepton has a tau as ancestor"),
        ),
    variables = cms.PSet(
        P4Vars,
        pdgId = Var("pdgId", int, doc="PDG id"), 
    )
)


tautagger = cms.EDProducer("GenJetTauTaggerProducer",
    src = rivetLeptonTable.src,
)

#rivetJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
#    src = cms.InputTag("particleLevel:jets"),
#    cut = cms.string(""),
#    name= cms.string("RivetJet"),
#    doc = cms.string("AK4 jets from Rivet-based ParticleLevelProducer"),
#    singleton = cms.bool(False), # the number of entries is variable
#    extension = cms.bool(False),
#    variables = cms.PSet(
#        # Identical to GenJets, so we just extend their flavor information
#        P4Vars,
#        hadronFlavour = Var("pdgId", int, doc="PDG id"),
#    )
#)

#rivetFatJetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
#    src = cms.InputTag("particleLevel:fatjets"),
#    cut = cms.string(""),
#    name= cms.string("GenFatJet"),
#    doc = cms.string("AK8 jets from Rivet-based ParticleLevelProducer"),
#    singleton = cms.bool(False), # the number of entries is variable
#    extension = cms.bool(False), # this is the main table
#    variables = cms.PSet(
#        P4Vars,
#        hadronFlavour = Var("pdgId", int, doc="PDG id"), 
#    )
#)

#rivetTagTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
#    src = cms.InputTag("particleLevel:tags"),
#    cut = cms.string(""),
#    name= cms.string("RivetTag"),
#    doc = cms.string("Tag particles from Rivet-based ParticleLevelProducer, momenta scaled down by 10e-20"),
#    singleton = cms.bool(False), # the number of entries is variable
#    extension = cms.bool(False), # this is the main table
#    variables = cms.PSet(
#        P4Vars,
#        pdgId = Var("pdgId", int, doc="PDG id"), 
#    )
#)

rivetMetTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("particleLevel:mets"),
    name = cms.string("MET"),
    doc = cms.string("MET from Rivet-based ParticleLevelProducer in fiducial volume abs(eta)<5"),
    singleton = cms.bool(True),  # there's always exactly one MET per event
    extension = cms.bool(True), # this is the main table
    variables = cms.PSet(
       fiducialGenPt  = Var("pt",  float, precision=10),
       fiducialGenPhi = Var("phi", float, precision=10),
    ),
)

HTXSCategoryTable = cms.EDProducer("SimpleHTXSFlatTableProducer",
    src = cms.InputTag("rivetProducerHTXS","HiggsClassification"),
    cut = cms.string(""),
    name = cms.string("HTXS"),
    doc = cms.string("HTXS classification"),
    singleton = cms.bool(True),
    extension = cms.bool(False),
    variables=cms.PSet(
        stage_0 = Var("stage0_cat",int, doc="HTXS stage-0 category"),
        stage_1_pTjet30 = Var("stage1_cat_pTjet30GeV",int, doc="HTXS stage-1 category (jet pt>30 GeV)"),
        stage_1_pTjet25 = Var("stage1_cat_pTjet25GeV",int, doc="HTXS stage-1 category (jet pt>25 GeV)"),
        stage1_1_cat_pTjet30GeV = Var("stage1_1_cat_pTjet30GeV",int,doc="HTXS stage-1.1 category(jet pt>30 GeV)"),
        stage1_1_cat_pTjet25GeV = Var("stage1_1_cat_pTjet25GeV",int,doc="HTXS stage-1.1 category(jet pt>25 GeV)"),
        stage1_1_fine_cat_pTjet30GeV = Var("stage1_1_fine_cat_pTjet30GeV",int,doc="HTXS stage-1.1-fine category(jet pt>30 GeV)"),
        stage1_1_fine_cat_pTjet25GeV = Var("stage1_1_fine_cat_pTjet25GeV",int,doc="HTXS stage-1.1-fine category(jet pt>25 GeV)"),
        Higgs_pt = Var("higgs.Pt()",float, doc="pt of the Higgs boson as identified in HTXS", precision=14),
        Higgs_y = Var("higgs.Rapidity()",float, doc="rapidity of the Higgs boson as identified in HTXS", precision=12),
        njets30 = Var("jets30.size()","uint8", doc="number of jets with pt>30 GeV as identified in HTXS"),
        njets25 = Var("jets25.size()","uint8", doc="number of jets with pt>25 GeV as identified in HTXS"),
   )
)


particleLevelSequence = cms.Sequence(mergedGenParticles + genParticles2HepMC + particleLevel + tautagger + genParticles2HepMCHiggsVtx + rivetProducerHTXS)
particleLevelTables = cms.Sequence(rivetLeptonTable + rivetMetTable + HTXSCategoryTable)
