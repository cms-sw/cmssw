from PhysicsTools.NanoAOD.taus_cff import *
<<<<<<< HEAD
from PhysicsTools.NanoAOD.jetMC_cff import *
from PhysicsTools.NanoAOD.globals_cff import genTable,genFilterTable
from PhysicsTools.NanoAOD.met_cff import metMCTable
=======
from PhysicsTools.NanoAOD.jets_cff import *
from PhysicsTools.NanoAOD.globals_cff import *
>>>>>>> 4cb142f8eb0 (New WeightGroups, improved parsing with helper class)
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.particlelevel_cff import *
from PhysicsTools.NanoAOD.lheInfoTable_cfi import *
from PhysicsTools.NanoAOD.genWeightsTable_cfi import *
from PhysicsTools.NanoAOD.lheWeightsTable_cfi import *

genWeights = cms.EDProducer("GenWeightProductProducer",
    genInfo = cms.InputTag("generator"),
    genLumiInfoHeader = cms.InputTag("generator"))

lheWeights = cms.EDProducer("LHEWeightProductProducer",
    lheSourceLabels = cms.vstring(["externalLHEProducer", "source"]),
    failIfInvalidXML = cms.untracked.bool(True)
    #lheWeightSourceLabels = cms.vstring(["externalLHEProducer", "source"])
)
'''
lheWeightsTable = cms.EDProducer(
    "LHEWeightsTableProducer",
    lheWeights = cms.VInputTag(["externalLHEProducer", "source", "lheWeights"]),
    lheWeightPrecision = cms.int32(14),
    genWeights = cms.InputTag("genWeights"),
    # Warning: you can use a full string, but only the first character is read.
    # Note also that the capitalization is important! For example, 'parton shower' 
    # must be lower case and 'PDF' must be capital
    weightgroups = cms.vstring(['scale', 'PDF', 'matrix element', 'unknown', 'parton shower']),
    # Max number of groups to store for each type above, -1 ==> store all found
    maxGroupsPerType = cms.vint32([-1, -1, -1, -1, 1]),
    # If empty or not specified, no critieria is applied to filter on LHAPDF IDs 
    #pdfIds = cms.untracked.vint32([91400, 306000, 260000]),
    #unknownOnlyIfEmpty = cms.untracked.vstring(['scale', 'PDF']),
    #unknownOnlyIfAllEmpty = cms.untracked.bool(False),
)
'''
nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

metGenTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("genMetTrue"),
    name = cms.string("GenMET"),
    doc = cms.string("Gen MET"),
    singleton = cms.bool(True),
    extension = cms.bool(False),
    variables = cms.PSet(
       pt  = Var("pt",  float, doc="pt", precision=10),
       phi = Var("phi", float, doc="phi", precision=10),
    ),
)

nanogenSequence = cms.Sequence(
    genWeights+
    lheWeights+
    nanoMetadata+
    particleLevel+
    genJetTable+
    patJetPartons+
    genJetFlavourAssociation+
    genJetFlavourTable+
    genJetAK8Table+
    genJetAK8FlavourAssociation+
    genJetAK8FlavourTable+
    tauGenJets+
    tauGenJetsSelectorAllHadrons+
    genVisTaus+
    genVisTauTable+
    genTable+
    genWeightsTable+
    lheWeightsTable+
    genParticleTables+
    tautagger+
    rivetProducerHTXS+
    particleLevelTables+
    metGenTable+
    lheInfoTable
)

nanogenMiniSequence = cms.Sequence(
    genWeights+
    lheWeights+
    nanoMetadata+
    mergedGenParticles+
    genParticles2HepMC+
    particleLevel+
    genJetTable+
    patJetPartons+
    genJetFlavourAssociation+
    genJetFlavourTable+
    genJetAK8Table+
    genJetAK8FlavourAssociation+
    genJetAK8FlavourTable+
    tauGenJets+
    tauGenJetsSelectorAllHadrons+
    genVisTaus+
    genVisTauTable+
    genTable+
    genWeightsTable+
    lheWeightsTable+
    genParticleTables+
    tautagger+
    genParticles2HepMCHiggsVtx+
    rivetProducerHTXS+
    particleLevelTables+
    metGenTable+
    lheInfoTable
)

def customizeNanoGENFromMini(process):
    # Why is this false by default?!
    process.lheInfoTable.storeLHEParticles = True
    process.genParticleTable.src = "prunedGenParticles"
    process.patJetPartons.particles = "prunedGenParticles"
    process.particleLevel.src = "genParticles2HepMC:unsmeared"
    process.rivetProducerHTXS.HepMCCollection = "genParticles2HepMCHiggsVtx:unsmeared"

    process.genJetTable.src = "slimmedGenJets"
    process.genJetFlavourAssociation.jets = process.genJetTable.src
    process.genJetFlavourTable.src = process.genJetTable.src
    process.genJetFlavourTable.jetFlavourInfos = "genJetFlavourAssociation"
    process.genJetAK8Table.src = "slimmedGenJetsAK8"
    process.genJetAK8FlavourAssociation.jets = process.genJetAK8Table.src
    process.genJetAK8FlavourTable.src = process.genJetAK8Table.src
    process.tauGenJets.GenParticles = "prunedGenParticles"
    process.genVisTaus.srcGenParticles = "prunedGenParticles"

    return process

def customizeNanoGEN(process):
    process.lheInfoTable.storeLHEParticles = True
    process.genParticleTable.src = "genParticles"
    process.patJetPartons.particles = "genParticles"
    process.particleLevel.src = "generatorSmeared"
    process.particleLevel.particleMaxEta = 999.
    process.particleLevel.lepMinPt = 0.
    process.particleLevel.lepMaxEta = 999.
    process.rivetProducerHTXS.HepMCCollection = "generatorSmeared"

    process.genJetTable.src = "ak4GenJets"
    process.genJetFlavourAssociation.jets = process.genJetTable.src
    process.genJetFlavourTable.src = process.genJetTable.src
    process.genJetFlavourTable.jetFlavourInfos = "genJetFlavourAssociation"
    process.genJetAK8Table.src = "ak8GenJets"
    process.genJetAK8FlavourAssociation.jets = process.genJetAK8Table.src
    process.genJetAK8FlavourTable.src = process.genJetAK8Table.src
    process.tauGenJets.GenParticles = "genParticles"
    process.genVisTaus.srcGenParticles = "genParticles"

    return process
