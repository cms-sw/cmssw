import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Herwig7Settings.Herwig7CH3TuneSettings_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7StableParticlesForDetector_cfi import *
from Configuration.Generator.Herwig7Settings.Herwig7MGMergingSettings_cfi import *


generator = cms.EDFilter("Herwig7GeneratorFilter",
    herwig7CH3SettingsBlock,
    herwig7StableParticlesForDetectorBlock,
    herwig7MGMergingSettingsBlock,
    configFiles = cms.vstring(),
    hw_user_settings = cms.vstring(
        'set FxFxHandler:MergeMode FxFx',
        'set FxFxHandler:njetsmax 2'
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

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/2017/13TeV/madgraph/V5_2.6.1/DYellell012j_5f_NLO_FXFX/dyellell012j_5f_NLO_FXFX_slc7_amd64_gcc700_CMSSW_10_6_4_tarball.tar.xz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    generateConcurrently = cms.untracked.bool(True),
    postGenerationCommand = cms.untracked.vstring('mergeLHE.py', '-i', 'thread*/cmsgrid_final.lhe', '-o', 'cmsgrid_final.lhe')
)


ProductionFilterSequence = cms.Sequence(generator)
