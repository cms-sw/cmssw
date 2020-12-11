import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")

process.source = cms.Source("EmptySource")

from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import * 
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosityParticles = cms.untracked.bool(True),
    comEnergy = cms.double(13000.),

    ExternalDecays = cms.PSet(
        EvtGen1 = cms.untracked.PSet(
            decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_NOLONGLIFE.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
            convertPythiaCodes = cms.untracked.bool(False), # this is needed since 1.6
            list_forced_decays = cms.vstring(),
            operates_on_particles = cms.vint32(0) #will decay all particles coded in interface, it test the whole system
        ),
        parameterSets = cms.vstring('EvtGen1')
    ),
        
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('Top:gg2ttbar = on',
                                        'Top:qqbar2ttbar = on'
                                        ),
        parameterSets = cms.vstring('processParameters')
    )
)

# The line below removes messages like "particle not recognized by pythia"
# It uses EvtGenSetting_cff above
process.generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(100)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(34)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Py8_tt_evtgen1.root')
)

process.genParticles.src = cms.InputTag("generator","unsmeared")

process.printGenParticles = cms.EDAnalyzer("ParticleListDrawer",
                                           src = cms.InputTag("genParticles"),
                                           maxEventsToPrint = cms.untracked.int32(3) )

process.p = cms.Path(process.generator*process.genParticles*process.printGenParticles)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
