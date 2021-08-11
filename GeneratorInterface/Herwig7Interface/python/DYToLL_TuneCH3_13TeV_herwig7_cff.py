import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Herwig7Settings.Herwig7StableParticlesForDetector_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7CH3TuneSettings_cfi import *


generator = cms.EDFilter("Herwig7GeneratorFilter",
    herwig7StableParticlesForDetectorBlock,
    herwig7CH3SettingsBlock,
    run = cms.string('InterfaceMatchboxTest'),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    configFiles = cms.vstring(),
    crossSection = cms.untracked.double(-1),
    pptoll = cms.vstring(
                'read snippets/PPCollider.in',
                'cd /Herwig/Generators',
                'set EventGenerator:EventHandler:LuminosityFunction:Energy 13000.0',
                'cd /Herwig/MatrixElements/',
                'insert SubProcess:MatrixElements[0] MEqq2gZ2ff'),
    parameterSets = cms.vstring('herwig7CH3PDF', 'herwig7CH3AlphaS', 'herwig7StableParticlesForDetector', 'pptoll'),
    filterEfficiency = cms.untracked.double(1.0),
)

ProductionFilterSequence = cms.Sequence(generator)
