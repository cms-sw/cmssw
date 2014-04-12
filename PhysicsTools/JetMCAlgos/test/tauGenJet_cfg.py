# event generation
# analysis of the gen event

import FWCore.ParameterSet.Config as cms


process = cms.Process("GEN")

process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")

# event generation ------------------------------------------------------

# single tau
process.load("Configuration.Generator.PythiaUESettings_cfi")
process.source = cms.Source("PythiaSource",
    Phimin = cms.untracked.double(0.0),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(15),
    Etamin = cms.untracked.double(-1.0),
    DoubleParticle = cms.untracked.bool(False),
    Phimax = cms.untracked.double(360.0),
    Ptmin = cms.untracked.double(50.0),
    Ptmax = cms.untracked.double(50.0001),
    Etamax = cms.untracked.double(1.0),
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        pythiaTauJets = cms.vstring('MDME(89,1)=0      ! no tau->electron', 
            'MDME(90,1)=0      ! no tau->muon'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'pythiaTauJets')
    )
)


# gen particles printouts -----------------------------------------------
# all this could go into a cff

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.particleListDrawer = cms.EDAnalyzer(
    "ParticleListDrawer",
    printOnlyHardInteraction = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(-1),
    src = cms.InputTag('genParticles')
  )

# path  -----------------------------------------------------------------


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)


process.p1 = cms.Path(
    process.genParticles *
    process.particleListDrawer *
    process.tauGenJets 
    )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('gen.root')
)
process.outpath = cms.EndPath(process.out)
