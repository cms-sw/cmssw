import FWCore.ParameterSet.Config as cms

from Configuration.Generator.DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV_cff import externalLHEProducer
from Configuration.Generator.Herwig7Settings.Herwig7CH3TuneSettings_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7StableParticlesForDetector_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7MGMergingSettings_cfi import *


generator = cms.EDFilter("Herwig7GeneratorFilter",
    herwig7CH3SettingsBlock,
    herwig7StableParticlesForDetectorBlock,
    herwig7MGMergingSettingsBlock,
    configFiles = cms.vstring(),
    hw_user_settings = cms.vstring(
        'set FxFxHandler:MergeMode TreeMG5',
        'set FxFxHandler:njetsmax 4'
    ),
    parameterSets = cms.vstring(
        'herwig7CH3PDF',
        'herwig7CH3AlphaS',
        'herwig7CH3MPISettings',
        'herwig7StableParticlesForDetector',
        'hw_mg_merging_settings',
        'hw_user_settings'
        ),
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
