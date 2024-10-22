import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/2017/13TeV/madgraph/V5_2.4.2/BulkGraviton_hh_GF_HH_narrow_M1000/v1/BulkGraviton_hh_GF_HH_narrow_M1000_slc6_amd64_gcc481_CMSSW_7_1_30_tarball.tar.xz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

generator = cms.EDFilter("Herwig7GeneratorFilter",
    run = cms.string('LHEinRun'),
    repository = cms.string('${HERWIGPATH}/HerwigDefaults.rpo'),
    dataLocation = cms.string('${HERWIGPATH:-6}'),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    configFiles = cms.vstring(),
    crossSection = cms.untracked.double(-1),
    parameterSets = cms.vstring("Matchbox"),
    filterEfficiency = cms.untracked.double(1.0),
    Matchbox = cms.vstring('read LHE.in'),
    runModeList = cms.untracked.string('read,run')
)

ProductionFilterSequence = cms.Sequence(generator)
