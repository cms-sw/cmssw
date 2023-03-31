import FWCore.ParameterSet.Config as cms

from Configuration.Generator.TTbar_Pow_LHE_13TeV_cff import externalLHEProducer
from Configuration.Generator.Herwig7Settings.Herwig7CH3TuneSettings_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7StableParticlesForDetector_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7LHECommonSettings_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7LHEPowhegSettings_cfi import *


generator = cms.EDFilter("Herwig7GeneratorFilter",
    herwig7CH3SettingsBlock,
    herwig7StableParticlesForDetectorBlock,
    herwig7LHECommonSettingsBlock,
    herwig7LHEPowhegSettingsBlock,
    configFiles = cms.vstring(),
    parameterSets = cms.vstring(
        'herwig7CH3PDF',
        'herwig7CH3AlphaS',
        'herwig7CH3MPISettings',
        'herwig7StableParticlesForDetector',
        'hw_lhe_common_settings',
        'hw_lhe_powheg_settings'),
    crossSection = cms.untracked.double(-1),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    filterEfficiency = cms.untracked.double(1.0),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    run = cms.string('InterfaceMatchboxTest'),
    runModeList = cms.untracked.string("read,run"),
    seed = cms.untracked.int32(12345)
)


ProductionFilterSequence = cms.Sequence(generator)
