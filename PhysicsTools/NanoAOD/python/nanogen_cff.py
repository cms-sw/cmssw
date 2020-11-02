from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.jets_cff import *
from PhysicsTools.NanoAOD.globals_cff import genTable
from PhysicsTools.NanoAOD.met_cff import metMCTable
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.particlelevel_cff import *
from PhysicsTools.NanoAOD.lheInfoTable_cfi import *
from PhysicsTools.NanoAOD.genWeightsTable_cfi import *
from PhysicsTools.NanoAOD.common_cff import Var,CandVars

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)

nanogenSequence = cms.Sequence(
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
    genParticleTables+
    tautagger+
    rivetProducerHTXS+
    particleLevelTables+
    metMCTable+
    genWeightsTable+
    lheInfoTable
)

def nanoGenCommonCustomize(process):
    process.rivetMetTable.extension = False
    process.lheInfoTable.storeLHEParticles = True
    process.lheInfoTable.precision = 14
    process.genWeightsTable.keepAllPSWeights = True
    process.genJetFlavourAssociation.jets = process.genJetTable.src
    process.genJetFlavourTable.src = process.genJetTable.src
    process.genJetAK8FlavourAssociation.jets = process.genJetAK8Table.src
    process.genJetAK8FlavourTable.src = process.genJetAK8Table.src
    process.particleLevel.particleMaxEta = 999.
    process.particleLevel.lepMinPt = 0.
    process.particleLevel.lepMaxEta = 999.
    process.genJetFlavourTable.jetFlavourInfos = "genJetFlavourAssociation"
    # Same as default RECO
    setGenPtPrecision(process, CandVars.pt.precision)
    setGenEtaPrecision(process, CandVars.eta.precision)
    setGenPhiPrecision(process, CandVars.phi.precision)

def customizeNanoGENFromMini(process):
    process.nanoAOD_step.insert(0, process.genParticles2HepMCHiggsVtx)
    process.nanoAOD_step.insert(0, process.genParticles2HepMC)
    process.nanoAOD_step.insert(0, process.mergedGenParticles)

    process.metMCTable.src = "slimmedMETs"
    process.metMCTable.variables.pt = Var("genMET.pt", float, doc="pt")
    process.metMCTable.variables.phi = Var("genMET.phi", float, doc="phi")
    process.metMCTable.variables.phi.precision = CandVars.phi.precision

    process.rivetProducerHTXS.HepMCCollection = "genParticles2HepMCHiggsVtx:unsmeared"
    process.genParticleTable.src = "prunedGenParticles"
    process.patJetPartons.particles = "prunedGenParticles"
    process.particleLevel.src = "genParticles2HepMC:unsmeared"

    process.genJetTable.src = "slimmedGenJets"
    process.genJetAK8Table.src = "slimmedGenJetsAK8"
    process.tauGenJets.GenParticles = "prunedGenParticles"
    process.genVisTaus.srcGenParticles = "prunedGenParticles"
    nanoGenCommonCustomize(process)

    return process

def customizeNanoGEN(process):
    process.metMCTable.src = "genMetTrue"
    process.metMCTable.variables = cms.PSet(PTVars)

    process.rivetProducerHTXS.HepMCCollection = "generatorSmeared"
    process.genParticleTable.src = "genParticles"
    process.patJetPartons.particles = "genParticles"
    process.particleLevel.src = "generatorSmeared"

    process.genJetTable.src = "ak4GenJets"
    process.genJetAK8Table.src = "ak8GenJets"
    process.tauGenJets.GenParticles = "genParticles"
    process.genVisTaus.srcGenParticles = "genParticles"

    # In case customizeNanoGENFromMini has already been called
    process.nanoAOD_step.remove(process.genParticles2HepMCHiggsVtx)
    process.nanoAOD_step.remove(process.genParticles2HepMC)
    process.nanoAOD_step.remove(process.mergedGenParticles)
    nanoGenCommonCustomize(process)
    return process

# Prune gen particles with tight conditions applied in usual NanoAOD
def pruneGenParticlesNano(process):
    process.finalGenParticles = finalGenParticles.clone()
    process.genParticleTable.src = "prunedGenParticles"
    process.patJetPartons.particles = "prunedGenParticles"
    process.nanoAOD_step.insert(0, process.finalGenParticles)
    return process

# Prune gen particles with conditions applied in usual MiniAOD
def pruneGenParticlesMini(process):
    from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import prunedGenParticles
    process.prunedGenParticles = prunedGenParticles.clone()
    if process.nanoAOD_step.contains(process.nanogenMiniSequence):
        raise ValueError("Applying the MiniAOD genParticle pruner to MiniAOD is redunant. " \
            "Use a different customization.")
    process.genParticleTable.src = "prunedGenParticles"
    process.patJetPartons.particles = "prunedGenParticles"
    process.nanoAOD_step.insert(0, process.prunedGenParticles)
    return process

def setGenFullPrecision(process):
    setGenPtPrecision(process, 23)
    setGenEtaPrecision(process, 23)
    setGenPhiPrecision(process, 23)

def setGenPtPrecision(process, precision):
    process.genParticleTable.variables.pt.precision = precision
    process.genJetTable.variables.pt.precision = precision
    process.metMCTable.variables.pt.precision = precision
    return process

def setGenEtaPrecision(process, precision):
    process.genParticleTable.variables.eta.precision = precision
    process.genJetTable.variables.eta.precision = precision
    return process

def setGenPhiPrecision(process, precision):
    process.genParticleTable.variables.phi.precision = precision
    process.genJetTable.variables.phi.precision = precision
    process.metMCTable.variables.phi.precision = precision
    return process

def setLHEFullPrecision(process):
    process.lheInfoTable.precision = 23
    return process

def setGenWeightsFullPrecision(process):
    process.genWeightsTable.lheWeightPrecision = 23
    return process
