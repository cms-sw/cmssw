from PhysicsTools.NanoAOD.taus_cff import *
from PhysicsTools.NanoAOD.jetMC_cff import *
from PhysicsTools.NanoAOD.globals_cff import genTable,genFilterTable
from PhysicsTools.NanoAOD.met_cff import metMCTable
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.particlelevel_cff import *
from PhysicsTools.NanoAOD.genWeightsTable_cfi import *
from PhysicsTools.NanoAOD.genVertex_cff import *
from PhysicsTools.NanoAOD.common_cff import Var,CandVars
from PhysicsTools.NanoAOD.simpleSingletonCandidateFlatTableProducer_cfi import simpleSingletonCandidateFlatTableProducer

nanoMetadata = cms.EDProducer("UniqueStringProducer",
    strings = cms.PSet(
        tag = cms.string("untagged"),
    )
)



nanogenSequence = cms.Sequence(
    nanoMetadata+
    cms.Sequence(particleLevelTask)+
    genJetTable+
    patJetPartonsNano+
    genJetFlavourAssociation+
    genJetFlavourTable+
    genSubJetAK8Table+
    genJetAK8Table+
    genJetAK8FlavourAssociation+
    genJetAK8FlavourTable+
    cms.Sequence(genTauTask)+
    genTable+
    genIso+
    genFilterTable+
    cms.Sequence(genParticleTablesTask)+
    cms.Sequence(genVertexTablesTask)+
    tautagger+
    rivetProducerHTXS+
    cms.Sequence(particleLevelTablesTask)+
    metMCTable+
    genWeightsTable
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
    setGenMassPrecision(process, CandVars.mass.precision)

    for output in ("NANOEDMAODSIMoutput", "NANOAODSIMoutput"):
        if hasattr(process, output):
            getattr(process, output).outputCommands.append("drop edmTriggerResults_*_*_*")

def customizeNanoGENFromMini(process):
    process.nanogenSequence.insert(0, process.genParticles2HepMCHiggsVtx)
    process.nanogenSequence.insert(0, process.genParticles2HepMC)
    process.nanogenSequence.insert(0, process.mergedGenParticles)

    process.metMCTable.src = "slimmedMETs"
    process.metMCTable.variables.pt = Var("genMET.pt", float, doc="pt")
    process.metMCTable.variables.phi = Var("genMET.phi", float, doc="phi")
    process.metMCTable.variables.phi.precision = CandVars.phi.precision

    process.rivetProducerHTXS.HepMCCollection = "genParticles2HepMCHiggsVtx:unsmeared"
    process.genParticleTable.src = "prunedGenParticles"
    process.patJetPartonsNano.particles = "prunedGenParticles"
    process.particleLevel.src = "genParticles2HepMC:unsmeared"
    process.genIso.genPart = "prunedGenParticles"

    process.genJetTable.src = "slimmedGenJets"
    process.genJetAK8Table.src = "slimmedGenJetsAK8"
    process.tauGenJetsForNano.GenParticles = "prunedGenParticles"
    process.genVisTaus.srcGenParticles = "prunedGenParticles"

    nanoGenCommonCustomize(process)

    return process

def customizeNanoGEN(process):
    process.metMCTable = simpleSingletonCandidateFlatTableProducer.clone(
        src = "genMetTrue",
        name = process.metMCTable.name,
        doc = process.metMCTable.doc,
        variables = cms.PSet(PTVars)
    )

    process.nanogenSequence.insert(0, process.genParticles2HepMCHiggsVtx)
    process.nanogenSequence.insert(0, process.genParticles2HepMC)
    process.genParticles2HepMCHiggsVtx.genParticles = "genParticles"
    process.genParticles2HepMC.genParticles = "genParticles"

    process.rivetProducerHTXS.HepMCCollection = "genParticles2HepMCHiggsVtx:unsmeared"
    process.genParticleTable.src = "genParticles"
    process.patJetPartonsNano.particles = "genParticles"
    process.particleLevel.src = "genParticles2HepMC:unsmeared"

    process.genJetTable.src = "ak4GenJetsNoNu"
    process.genJetAK8Table.src = "ak8GenJetsNoNu"
    process.tauGenJetsForNano.GenParticles = "genParticles"
    process.genVisTaus.srcGenParticles = "genParticles"
    process.load("RecoJets.JetProducers.ak8GenJets_cfi")
    process.ak8GenJetsNoNuConstituents =  process.ak8GenJetsConstituents.clone(src='ak8GenJetsNoNu')
    process.ak8GenJetsNoNuSoftDrop = process.ak8GenJetsSoftDrop.clone(src=cms.InputTag('ak8GenJetsNoNuConstituents', 'constituents'))
    process.genSubJetAK8Table.src = "ak8GenJetsNoNuSoftDrop"
    process.nanogenSequence.insert(0, process.ak8GenJetsNoNuSoftDrop)
    process.nanogenSequence.insert(0, process.ak8GenJetsNoNuConstituents)
    # In case customizeNanoGENFromMini has already been called
    process.nanogenSequence.remove(process.mergedGenParticles)
    process.nanogenSequence.remove(process.genIso)
    delattr(process.genParticleTable.externalVariables,"iso")
    nanoGenCommonCustomize(process)
    return process

# Prune gen particles with tight conditions applied in usual NanoAOD
def pruneGenParticlesNano(process):
    process.finalGenParticles.src = process.genParticleTable.src.getModuleLabel()
    process.genParticleTable.src = "finalGenParticles"
    process.nanogenSequence.insert(0, process.finalGenParticles)
    return process

# Prune gen particles with conditions applied in usual MiniAOD
def pruneGenParticlesMini(process):
    if process.nanogenSequence.contains(process.mergedGenParticles):
        raise ValueError("Applying the MiniAOD genParticle pruner to MiniAOD is redunant. " \
            "Use a different customization.")
    from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import prunedGenParticles
    process.prunedGenParticles = prunedGenParticles.clone()
    process.prunedGenParticles.src = "genParticles"
    process.genParticleTable.src = "prunedGenParticles"

    process.nanogenSequence.insert(0, process.prunedGenParticles)
    return process

def setGenFullPrecision(process):
    process = setGenPtPrecision(process, 23)
    process = setGenEtaPrecision(process, 23)
    process = setGenPhiPrecision(process, 23)
    process = setGenMassPrecision(process, 23)
    return process

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

def setGenMassPrecision(process, precision):
    process.genParticleTable.variables.mass.precision = precision
    process.genJetTable.variables.mass.precision = precision
    return process

def setLHEFullPrecision(process):
    process.lheInfoTable.precision = 23
    return process

def setGenWeightsFullPrecision(process):
    process.genWeightsTable.lheWeightPrecision = 23
    return process
